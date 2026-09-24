"""Shared loaders and chemistry utilities for the bond-enthalpy domain analysis.

Every module in this run imports from here so that the definition of the
analysis unit, the bond-class labeler, and the provenance joins are identical
across the parallel workstreams.

Evidence boundary: only files under ``data/`` are read.  No network access.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OUT = PROJECT / "output"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"

TSV = DATA / "normalized_views" / "experimental_benchmark.tsv"
DB_GZ = DATA / "source_data" / "bde_db_full.csv.gz"
XLSX = DATA / "source_data" / "experimental_source_tables.xlsx"

CHANNELS = ("quantum", "learned", "reference")
CHAN_COL = {c: f"{c}_estimate_bde_kcal_mol" for c in CHANNELS}
EXP_COL = "experimental_bde_kcal_mol"

# Element priority used when naming a bond class from its two radical centres.
_ELEMENT_ORDER = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# chemistry helpers
# --------------------------------------------------------------------------

def _rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def radical_center(smiles: str):
    """Element symbol of the radical-bearing (or sole) atom of a fragment.

    Sanitised parsing is essential: with ``sanitize=False`` RDKit reports no
    radical electrons at all for these open-shell SMILES.
    """
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    radicals = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    atom = radicals[0] if len(radicals) == 1 else mol.GetAtomWithIdx(0)
    return {
        "element": atom.GetSymbol(),
        "aromatic": bool(atom.GetIsAromatic()),
        "in_ring": bool(atom.IsInRing()),
        "n_radical_atoms": len(radicals),
    }


def bond_class(frag_a: str, frag_b: str) -> str | None:
    """Bond class label, e.g. ``C-H``, from the two fragment radical centres."""
    ra, rb = radical_center(frag_a), radical_center(frag_b)
    if ra is None or rb is None:
        return None
    els = sorted(
        [ra["element"], rb["element"]],
        key=lambda e: _ELEMENT_ORDER.index(e) if e in _ELEMENT_ORDER else 99,
    )
    return "-".join(els)


def heavy_atom_count(smiles: str) -> int | None:
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else mol.GetNumHeavyAtoms()


def molecular_formula(smiles: str) -> str | None:
    Chem = _rdkit()
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


def element_set(smiles: str) -> frozenset:
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return frozenset()
    return frozenset(a.GetSymbol() for a in mol.GetAtoms())


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

def load_benchmark() -> pd.DataFrame:
    """The 543-observation experimental benchmark view (TSV)."""
    return pd.read_csv(TSV, sep="\t")


def load_xlsx() -> pd.DataFrame:
    """The experimental source workbook, read directly with openpyxl.

    ``pandas.read_excel`` is not usable here: the installed pandas requires
    openpyxl >= 3.1.5 while the environment provides 3.0.9.  Reading through
    openpyxl directly keeps the workbook inside the evidence boundary.
    """
    import openpyxl

    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["Sheet_001"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    frame = pd.DataFrame(rows[1:], columns=header)
    wb.close()
    return frame


def load_db() -> pd.DataFrame:
    """The complete row-level BDE database."""
    return pd.read_csv(DB_GZ)


# --------------------------------------------------------------------------
# derived analysis table
# --------------------------------------------------------------------------

def build_analysis_table(verbose: bool = True, use_cache: bool = True) -> pd.DataFrame:
    """Benchmark view joined to the database and enriched with descriptors.

    Adds:
      bond_class      chemical bond class from the fragment radical centres
      db_bt           bond class recorded in the database (matched rows only)
      labeler_ok      whether our labeler reproduces db_bt
      in_db_row       exact (molecule, fragment1, fragment2) row exists in DB
      mol_in_db       parent molecule exists anywhere in DB
      db_bde          database bond enthalpy on matched rows (the quantum channel)
      hac_parent      heavy-atom count of the parent molecule
      train_flag      ``in_reference_training_set`` from the source workbook
      per-channel signed error and absolute error versus experiment

    The enriched table is cached to ``results/analysis_table.csv`` because
    the RDKit pass over 419 parent molecules takes several minutes; the cache is
    keyed on the sha256 of the three source files, so a changed input
    invalidates it automatically.
    """
    cache = RESULTS / "analysis_table.csv"
    cache_key = RESULTS / "analysis_table.key"
    key = "-".join(sha256(p) for p in (TSV, DB_GZ, XLSX))
    if use_cache and cache.exists() and cache_key.exists():
        if cache_key.read_text().strip() == key:
            frame = pd.read_csv(cache)
            if verbose:
                print(f"[common] loaded cached analysis table {frame.shape} from {cache}")
            return frame

    bench = load_benchmark()
    xlsx = load_xlsx()
    db = load_db()

    # ---- provenance join -------------------------------------------------
    db_rows = {}
    for m, f1, f2, bt, bde in zip(
        db.molecule, db.fragment1, db.fragment2, db.bond_type, db.bde
    ):
        db_rows[(m, f1, f2)] = (bt, bde)
    db_molecules = set(db.molecule)
    db_pair_set = set(zip(db.molecule, db.fragment1, db.fragment2))

    keys = list(
        zip(
            bench.parent_structure_smiles,
            bench.fragment_a_smiles,
            bench.fragment_b_smiles,
        )
    )
    bench["db_bt"] = [db_rows[k][0] if k in db_rows else None for k in keys]
    bench["db_bde"] = [db_rows[k][1] if k in db_rows else np.nan for k in keys]
    bench["in_db_row"] = [k in db_pair_set for k in keys]
    bench["mol_in_db"] = bench.parent_structure_smiles.isin(db_molecules)

    # ---- chemistry -------------------------------------------------------
    bench["bond_class"] = [
        bond_class(a, b) for a, b in zip(bench.fragment_a_smiles, bench.fragment_b_smiles)
    ]
    bench["labeler_ok"] = np.where(
        bench.db_bt.notna(), bench.bond_class == bench.db_bt, np.nan
    )
    hac_cache: dict[str, int | None] = {}
    for m in pd.unique(bench.parent_structure_smiles):
        hac_cache[m] = heavy_atom_count(m)
    bench["hac_parent"] = bench.parent_structure_smiles.map(hac_cache)

    # ---- workbook cross-check -------------------------------------------
    xlsx = xlsx.rename(
        columns={
            "Molecule": "parent_structure_smiles",
            "Radical 1": "fragment_a_smiles",
            "Radical 2": "fragment_b_smiles",
        }
    )
    for col, new in (
        ("observed_value", "xlsx_observed"),
        ("estimate_quantum", "xlsx_quantum"),
        ("estimate_learned", "xlsx_learned"),
        ("estimate_reference", "xlsx_reference"),
        ("in_reference_training_set", "train_flag"),
    ):
        xlsx[new] = xlsx[col]
    bench = bench.merge(
        xlsx[
            [
                "parent_structure_smiles",
                "fragment_a_smiles",
                "fragment_b_smiles",
                "xlsx_observed",
                "xlsx_quantum",
                "xlsx_learned",
                "xlsx_reference",
                "train_flag",
            ]
        ],
        on=["parent_structure_smiles", "fragment_a_smiles", "fragment_b_smiles"],
        how="left",
        validate="one_to_one",
    )

    # ---- errors ----------------------------------------------------------
    exp = bench[EXP_COL].to_numpy()
    for c in CHANNELS:
        pred = bench[CHAN_COL[c]].to_numpy()
        bench[f"{c}_err"] = pred - exp
        bench[f"{c}_ae"] = np.abs(pred - exp)

    # database gap against the two calibrated channels
    bench["db_minus_quantum"] = bench.db_bde - bench.quantum_estimate_bde_kcal_mol
    bench["db_minus_learned"] = bench.db_bde - bench.learned_estimate_bde_kcal_mol

    if verbose:
        print(
            f"[common] benchmark rows={len(bench)} "
            f"labeler_agreement={int(bench.labeler_ok.sum())}/"
            f"{int(bench.labeler_ok.notna().sum())} "
            f"in_db_row={int(bench.in_db_row.sum())}"
        )
    if use_cache:
        RESULTS.mkdir(parents=True, exist_ok=True)
        bench.to_csv(cache, index=False)
        cache_key.write_text(key, encoding="utf-8")
        if verbose:
            print(f"[common] cached analysis table to {cache}")
    return bench


# --------------------------------------------------------------------------
# statistics helpers (no scipy dependency for the core numbers)
# --------------------------------------------------------------------------

def paired_bootstrap_ci(
    diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 20260919
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of a paired difference."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(diff.mean()), float(lo), float(hi)


def paired_t_test(diff: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t statistic and p-value (Student t, df = n-1)."""
    from scipy import stats

    diff = np.asarray(diff, dtype=float)
    t, p = stats.ttest_1samp(diff, 0.0)
    return float(t), float(p)


def wilcoxon(diff: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    diff = np.asarray(diff, dtype=float)
    res = stats.wilcoxon(diff, zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size between two independent samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(x, dtype=float))))


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(math.sqrt(float(np.mean(x**2))))


def save_json(obj, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"[common] wrote {path}")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")


def dataframe_records(df: pd.DataFrame) -> list[dict]:
    """Records with NaN converted to None so the JSON stays valid."""
    return json.loads(df.to_json(orient="records"))