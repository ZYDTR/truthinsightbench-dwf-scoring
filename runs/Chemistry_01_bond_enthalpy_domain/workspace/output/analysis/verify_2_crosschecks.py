"""Adversarial cross-checks for claim 2, using a SECOND independent method.

Method A (in verify_2_coverage.py): RDKit GetNumHeavyAtoms on parsed parents.
Method B (here): (i) count heavy atoms from rdMolDescriptors.CalcMolFormula,
(ii) raw-string element scan of the SMILES, (iii) row-level filter for the
"beyond HAC 10" count vs unique-molecule mapping.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

P = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
bench = pd.read_csv(P / "data/normalized_views/experimental_benchmark.tsv", sep="\t")
db = pd.read_csv(P / "data/source_data/bde_db_full.csv.gz")

res: dict = {}
rep: list[str] = []


def p(*a):
    s = " ".join(str(x) for x in a)
    rep.append(s)
    print(s, flush=True)


t0 = time.time()

# ---- (i) HAC via molecular formula (independent of GetNumHeavyAtoms) ------
# formula like 'C7H12N2O'; heavy atoms = C,N,O,S,... counts, excluding H
FORM = re.compile(r"([A-Z][a-z]?)(\d*)")
uniq = pd.unique(db["molecule"].astype(str))
hac_form = {}
bad = []
for m in uniq:
    mol = Chem.MolFromSmiles(m)
    f = rdMolDescriptors.CalcMolFormula(mol)
    f = f.split(".")[0].replace("+", "").replace("-", "")
    heavy = 0
    for el, n in FORM.findall(f):
        if el == "H":
            continue
        heavy += int(n) if n else 1
    hac_form[m] = heavy
hv = list(hac_form.values())
p(f"B) HAC via CalcMolFormula over {len(uniq)} unique db molecules: "
  f"min={min(hv)} max={max(hv)}")
res["b_hac_max_formula"] = max(hv)
res["b_hac_max_formula_agree_with_GetNumHeavyAtoms"] = (max(hv) == 10)
# directly compare per-molecule
mismatch = 0
for m in uniq:
    if hac_form[m] != Chem.MolFromSmiles(m).GetNumHeavyAtoms():
        mismatch += 1
p(f"B) per-molecule HAC mismatches between formula-count and GetNumHeavyAtoms = {mismatch}")
res["b_per_molecule_hac_mismatches"] = mismatch
res["b_hac_max_molecule_examples"] = sorted(
    [m for m in uniq if hac_form[m] == max(hv)], key=len)[:5]

# ---- (ii) raw-string element scan (no RDKit) ------------------------------
TOKEN = re.compile(r"\[([A-Za-z@+\-0-9H]+)\]|Br|Cl|Si|Se|B|C|N|O|S|P|F|I")
NOBLE = {"Br", "Cl", "Si", "Se", "B", "C", "N", "O", "S", "P", "F", "I"}


def raw_elements(smi: str) -> set:
    out = set()
    i = 0
    while i < len(smi):
        c = smi[i]
        if c == "[":
            j = smi.index("]", i)
            body = smi[i + 1 : j]
            # atom symbol is leading element letters before H or digit or @ or +-
            mm = re.match(r"([A-Z][a-z]?)", body)
            if mm:
                out.add(mm.group(1))
            i = j + 1
        elif c.isupper() and c != "H":
            two = smi[i : i + 2]
            if two in NOBLE:
                out.add(two)
                i += 2
            else:
                out.add(c)
                i += 1
        else:
            i += 1
    return out


raw_mol_el = Counter()
for m in uniq:
    for e in raw_elements(m):
        raw_mol_el[e] += 1
p(f"B) raw-string element scan over unique db parent SMILES = "
  f"{dict(sorted(raw_mol_el.items()))}")
res["b_raw_element_scan_unique_parents"] = dict(sorted(raw_mol_el.items()))
# row weighted raw scan
raw_row_el = Counter()
for m in db["molecule"].astype(str):
    for e in raw_elements(m):
        raw_row_el[e] += 1
res["b_raw_element_scan_raw_rows"] = dict(sorted(raw_row_el.items()))
p(f"B) raw-string scan row-weighted = {dict(sorted(raw_row_el.items()))}")

# any explicit [H] in parent molecule column?
n_explicit_h = sum(1 for m in uniq if "[H]" in m or "H]" in m)
p(f"B) unique db parents containing an explicit H token in SMILES: {n_explicit_h}")
res["b_unique_parents_with_explicit_H"] = n_explicit_h

# ---- (iii) beyond-HAC-10 count: row filter vs unique-molecule mapping -----
# method 1: row-level filter (per observation), HAC of its parent
bench_hac = {}
for m in pd.unique(bench["parent_structure_smiles"].astype(str)):
    mo = Chem.MolFromSmiles(m)
    bench_hac[m] = mo.GetNumHeavyAtoms() if mo else None
def cutoff_formula(series):
    return sum(1 for m in series if bench_hac[m] is not None and bench_hac[m] > 10)

row_filter = sum(
    1 for m in bench["parent_structure_smiles"].astype(str)
    if bench_hac[m] is not None and bench_hac[m] > 10
)
# method 2: unique molecules above cutoff, then observations via groupby
mols_above = {m for m, h in bench_hac.items() if h is not None and h > 10}
uniq_map = bench["parent_structure_smiles"].astype(str).isin(mols_above).sum()
# method 3: numeric compare via vectorised pandas Series
bs = bench["parent_structure_smiles"].astype(str).map(bench_hac)
vec = int((bs > 10).sum())
p(f"B) beyond-HAC-10: row-filter={row_filter}  unique-mol-membership={uniq_map}  "
  f"vectorised={vec}  (agree={row_filter == uniq_map == vec})")
res["b_beyond_hac10_row_filter"] = row_filter
res["b_beyond_hac10_unique_membership"] = uniq_map
res["b_beyond_hac10_vectorised"] = vec
res["b_beyond_hac10_methods_agree"] = (row_filter == uniq_map == vec)
res["b_n_unique_benchmark_molecules_above_10"] = len(mols_above)

# ---- duplicate / unit checks on the benchmark ----------------------------
trip = list(zip(bench["parent_structure_smiles"], bench["fragment_a_smiles"],
                bench["fragment_b_smiles"]))
p(f"B) benchmark observations={len(trip)}  unique (mol,f1,f2) triples={len(set(trip))}")
res["b_benchmark_obs"] = len(trip)
res["b_benchmark_unique_triples"] = len(set(trip))
res["b_benchmark_unique_frag_pairs"] = len(
    set(zip(bench["fragment_a_smiles"], bench["fragment_b_smiles"])))
res["b_benchmark_unique_molecules"] = int(
    bench["parent_structure_smiles"].nunique())
p(f"B) unique fragment pairs={res['b_benchmark_unique_frag_pairs']}  "
  f"unique parent molecules={res['b_benchmark_unique_molecules']}")

# ?? could any benchmark triple match a db row *ignoring* stereo markers? ---
def strip_stereo(s):
    return re.sub(r"[/\\@]", "", s)


db_triples = set(zip(db["molecule"], db["fragment1"], db["fragment2"]))
db_triples_nostereo = set(
    (strip_stereo(a), strip_stereo(b), strip_stereo(c))
    for a, b, c in db_triples)
n1 = sum(k in db_triples for k in trip)
n2 = sum((strip_stereo(a), strip_stereo(b), strip_stereo(c)) in db_triples_nostereo
         for a, b, c in trip)
p(f"B) exact triples strict={n1}  stereo-insensitive={n2}")
res["b_exact_strict"] = n1
res["b_exact_stereo_insensitive"] = n2

# parent present, stereo-insensitive
dbmol_ns = {strip_stereo(m) for m in db["molecule"].astype(str)}
pm = sum(strip_stereo(m) in dbmol_ns for m in bench["parent_structure_smiles"].astype(str))
p(f"B) parent present strict={int(bench['parent_structure_smiles'].isin(set(db['molecule'].astype(str))).sum())} "
  f"stereo-insensitive={pm}")
res["b_parent_present_stereo_insensitive"] = pm

# ---- (iv) workbook cross-check on the benchmark boundary ------------------
import openpyxl

wb = openpyxl.load_workbook(P / "data/source_data/experimental_source_tables.xlsx",
                            read_only=True, data_only=True)
p(f"B) workbook sheets = {wb.sheetnames}")
for sh in wb.sheetnames:
    ws = wb[sh]
    p(f"   sheet {sh}: {ws.max_row} rows x {ws.max_column} cols")
res["b_workbook_sheets"] = wb.sheetnames
ws = wb["Sheet_001"]
rows = list(ws.iter_rows(values_only=True))
hdr = list(rows[0])
p(f"B) Sheet_001 header = {hdr}")
res["b_sheet_header"] = [str(h) for h in hdr]
wdf = pd.DataFrame(rows[1:], columns=hdr)
p(f"B) workbook rows={len(wdf)}  cols={list(wdf.columns)}")
res["b_workbook_rows"] = len(wdf)
col = {c.lower(): c for c in wdf.columns}
mcol = col.get("molecule")
if mcol:
    mo_set = pd.unique(wdf[mcol].astype(str))
    w_hac = {}
    for m in mo_set:
        mo = Chem.MolFromSmiles(m)
        w_hac[m] = mo.GetNumHeavyAtoms() if mo else None
    wh = pd.Series([w_hac[m] for m in mo_set])
    p(f"B) workbook unique molecules={len(mo_set)} HAC max={wh.max()} "
      f"n>10={int((wh>10).sum())}")
    res["b_workbook_unique_molecules"] = len(mo_set)
    res["b_workbook_hac_max"] = int(wh.max())
    res["b_workbook_molecules_gt10"] = int((wh > 10).sum())
    wels = Counter()
    for m in mo_set:
        for e in (raw_elements(m) or ()):
            wels[e] += 1
    res["b_workbook_raw_elements"] = dict(sorted(wels.items()))
    p(f"B) workbook element scan = {dict(sorted(wels.items()))}")
wb.close()

(Path(__file__).with_suffix(".json")).write_text(json.dumps(
    {"generated_by": "verify_2_crosschecks.py", "crosschecks": res, "log": rep},
    indent=2, default=str))
p(f"wrote crosscheck json ({time.time()-t0:.1f}s)")