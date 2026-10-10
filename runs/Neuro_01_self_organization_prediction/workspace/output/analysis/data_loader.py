"""Shared data access for Neuro_01_self_organization_prediction.

Loads all six condition MAT files into a flat list of trials.
Analysis unit = ONE complete trial. Time points within a trial are NOT
independent replicates (see data_manifest.json warnings).

Each returned trial is a dict:
    condition   : str      (apv, bic, ctrl, dzp, mix0, mix50)
    file        : str
    trial_index : int      (0-based within the file)
    s           : ndarray (25600, 2)    hidden sources
    o           : ndarray (25600, 32)   sensory stimulus
    r           : ndarray (25600, 64)   neural response
"""
import glob
import os

import numpy as np
import scipy.io as sio

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw")


def _first_array(obj):
    """Take the first (and only) numeric array out of a scalar object cell."""
    arr = np.asarray(obj)
    if arr.dtype == object:
        return arr.flat[0]
    return arr


def _load_condition(f):
    cond = os.path.basename(f).split("response_data_")[-1].split(".mat")[0]
    d = sio.loadmat(f)
    var_name = next(k for k in d.keys() if not k.startswith("__"))
    cell = d[var_name].reshape(-1)  # (N,1) -> (N,)
    trials = []
    for i in range(cell.size):
        rec = cell[i]
        trials.append({
            "condition": cond,
            "file": os.path.basename(f),
            "trial_index": i,
            "s": np.asarray(_first_array(rec["s"]), dtype=float),
            "o": np.asarray(_first_array(rec["o"]), dtype=float),
            "r": np.asarray(_first_array(rec["r"]), dtype=float),
        })
    return trials


def load_all(base_dir=BASE_DIR):
    """Load every trial (memory-heavy: ~1.2 GB). Use load_condition for bounded work."""
    trials = []
    for f in sorted(glob.glob(os.path.join(base_dir, "response_data_*.mat"))):
        trials.extend(_load_condition(f))
    return trials


def load_conditions(base_dir=BASE_DIR, conditions=None):
    """Load a subset of conditions. Returns dict condition -> list of trials.
    Bounded memory: only the requested conditions are held at once."""
    conditions = conditions or sorted(set(t["condition"] for t in load_all(base_dir)))
    out = {}
    for f in sorted(glob.glob(os.path.join(base_dir, "response_data_*.mat"))):
        cond = os.path.basename(f).split("response_data_")[-1].split(".mat")[0]
        if cond in conditions:
            out[cond] = _load_condition(f)
    return out


def trial_condition_map(trials):
    conds = {}
    for t in trials:
        conds.setdefault(t["condition"], []).append(t["trial_index"])
    return conds


if __name__ == "__main__":
    ts = load_all()
    from collections import Counter
    print("total trials:", len(ts))
    print("by condition:", dict(Counter(t["condition"] for t in ts)))
    t = ts[0]
    print("first trial:", t["condition"], t["trial_index"],
          "s", t["s"].shape, "o", t["o"].shape, "r", t["r"].shape)
    # verify all shapes consistent
    shapes = {(t["s"].shape, t["o"].shape, t["r"].shape) for t in ts}
    print("unique shapes:", shapes)
    nans = sum(np.isnan(t[k]).any() for t in ts for k in ("s", "o", "r"))
    print("any NaN:", nans)
