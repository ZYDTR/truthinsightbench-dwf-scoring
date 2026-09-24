"""WS4 - Failure modes, outliers and the hard boundary of each channel.

Every number reported by this workstream is produced here.  Reads only the
workspace data via ``common``.
"""
from __future__ import annotations

import sys
import json
import math

sys.path.insert(0, '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')

import numpy as np
import pandas as pd

import common
from common import CHANNELS

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors

RDLogger.DisableLog("rdApp.*")

PROJECT = '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project'

# ---------------------------------------------------------------------------
# descriptor block
# ---------------------------------------------------------------------------

_DESC_CACHE: dict[str, dict] = {}


def descriptors(smiles: str) -> dict:
    if smiles in _DESC_CACHE:
        return _DESC_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out = dict(parse_ok=False)
    else:
        ring_info = mol.GetRingInfo()
        arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        out = dict(
            parse_ok=True,
            hac=int(mol.GetNumHeavyAtoms()),
            n_rings=int(rdMolDescriptors.CalcNumRings(mol)),
            n_arom_rings=int(arom_rings),
            n_rotb=int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            n_bridge=int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
            n_spiro=int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
            n_fused=int(ring_info.NumRings() - rdMolDescriptors.CalcNumSpiroAtoms(mol)
                        - rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
            n_hetero=int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            fsp3=float(rdMolDescriptors.CalcFractionCSP3(mol)),
            formula=rdMolDescriptors.CalcMolFormula(mol),
            elem_set=",".join(sorted({a.GetSymbol() for a in mol.GetAtoms()})),
        )
    _DESC_CACHE[smiles] = out
    return out


def frag_radical_flags(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(parse_ok=False)
    rads = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if not rads:
        return dict(parse_ok=False)
    a = rads[0]
    return dict(
        parse_ok=True,
        el=a.GetSymbol(),
        arom=bool(a.GetIsAromatic()),
        in_ring=bool(a.IsInRing()),
        n_rad=len(rads),
        n_heavy=int(mol.GetNumHeavyAtoms()),
        ring_sizes=[len(r) for r in mol.GetRingInfo().AtomRings() if a.GetIdx() in r],
    )


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parent_cache = {s: descriptors(s) for s in pd.unique(df.parent_structure_smiles)}
    for k in ("hac", "n_rings", "n_arom_rings", "n_rotb", "n_bridge", "n_spiro",
              "n_fused", "n_hetero", "fsp3", "formula", "elem_set"):
        df["p_" + k] = df.parent_structure_smiles.map(
            lambda s, k=k: parent_cache[s].get(k) if parent_cache[s].get("parse_ok") else np.nan
        )
    fa = {s: frag_radical_flags(s) for s in pd.unique(df.fragment_a_smiles)}
    fb = {s: frag_radical_flags(s) for s in pd.unique(df.fragment_b_smiles)}
    df["fa_el"] = df.fragment_a_smiles.map(lambda s: fa[s].get("el"))
    df["fb_el"] = df.fragment_b_smiles.map(lambda s: fb[s].get("el"))
    df["fa_arom"] = df.fragment_a_smiles.map(lambda s: fa[s].get("arom"))
    df["fb_arom"] = df.fragment_b_smiles.map(lambda s: fb[s].get("arom"))
    df["fa_in_ring"] = df.fragment_a_smiles.map(lambda s: fa[s].get("in_ring"))
    df["fb_in_ring"] = df.fragment_b_smiles.map(lambda s: fb[s].get("in_ring"))
    df["fa_frag_hac"] = df.fragment_a_smiles.map(lambda s: fa[s].get("n_heavy"))
    df["fb_frag_hac"] = df.fragment_b_smiles.map(lambda s: fb[s].get("n_heavy"))
    # identical fragments -> fully symmetric bond
    df["same_frag"] = df.fragment_a_smiles == df.fragment_b_smiles
    df["both_rad_in_ring"] = df.fa_in_ring.astype(bool) & df.fb_in_ring.astype(bool)
    df["any_rad_arom"] = df.fa_arom.astype(bool) | df.fb_arom.astype(bool)
    # crowded / steric flag requested by WS4
    df["crowded"] = df.both_rad_in_ring | (df.p_n_rings >= 3)
    # the two fragments are constitutionally identical (canonical comparison)
    canon = {}
    for s in set(df.fragment_a_smiles) | set(df.fragment_b_smiles):
        m = Chem.MolFromSmiles(s)
        canon[s] = Chem.MolToSmiles(m) if m is not None else None
    df["same_frag_canon"] = [
        canon[a] is not None and canon[a] == canon[b]
        for a, b in zip(df.fragment_a_smiles, df.fragment_b_smiles)
    ]
    return df


# ---------------------------------------------------------------------------
# cached enriched table (common.build_analysis_table takes ~6 min; cache it)
# ---------------------------------------------------------------------------

from pathlib import Path

CACHE = Path(PROJECT) / "output" / "ws4_analysis_table.pkl"


def analysis_table(rebuild: bool = False) -> pd.DataFrame:
    """Enriched 543-row table, memoised to disk because the shared builder is slow."""
    if CACHE.exists() and not rebuild:
        return pd.read_pickle(CACHE)
    df = enrich(common.build_analysis_table(verbose=False))
    df.to_pickle(CACHE)
    return df
