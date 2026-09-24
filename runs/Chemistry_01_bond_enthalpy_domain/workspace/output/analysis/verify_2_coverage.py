"""Independent adversarial re-computation of the database-coverage claim (claim 2).

Reads ONLY the raw files under data/.  Does not import the original workstream
scripts (no ws*.py); only rdkit / pandas / numpy from the environment.
Every unique SMILES string is parsed with RDKit exactly once and cached.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT = Path(
    "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project"
)
TSV = PROJECT / "data" / "normalized_views" / "experimental_benchmark.tsv"
DB_GZ = PROJECT / "data" / "source_data" / "bde_db_full.csv.gz"

T0 = time.time()
out: dict = {}
report: list[str] = []


def p(*a):
    line = " ".join(str(x) for x in a)
    report.append(line)
    print(f"[{time.time()-T0:7.1f}s] {line}", flush=True)


_CACHE: dict[str, tuple[int | None, str | None]] = {}


def feat(smi: str):
    """(heavy-atom count, sorted element string) for a SMILES, cached."""
    if smi in _CACHE:
        return _CACHE[smi]
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        res = (None, None)
    else:
        res = (mol.GetNumHeavyAtoms(),
               ",".join(sorted({a.GetSymbol() for a in mol.GetAtoms()})))
    _CACHE[smi] = res
    return res


def hac(smi):
    return feat(smi)[0]


def elset(smi):
    e = feat(smi)[1]
    return None if e is None else set(e.split(","))


# ---------------------------------------------------------------- load raw
bench = pd.read_csv(TSV, sep="\t")
db = pd.read_csv(DB_GZ)
p(f"benchmark rows={len(bench)}")
p(f"db rows={len(db)} cols={list(db.columns)}")

db_mol = db["molecule"].astype(str)
db_f1 = db["fragment1"].astype(str)
db_f2 = db["fragment2"].astype(str)
b_mol = bench["parent_structure_smiles"].astype(str)
b_f1 = bench["fragment_a_smiles"].astype(str)
b_f2 = bench["fragment_b_smiles"].astype(str)

uniq_mol = pd.unique(db_mol)
p(f"db unique molecule strings = {len(uniq_mol)}  (claim: 42577)")
for i, m in enumerate(uniq_mol):
    feat(m)
    if i and i % 20000 == 0:
        p(f"  parsed {i}/{len(uniq_mol)} db molecules")
p(f"parsed all db molecules; unparsable = {sum(1 for m in uniq_mol if hac(m) is None)}")

hv = [hac(m) for m in uniq_mol]
HAC_MAX = max(hv)
p(f"DB UNIQUE-MOLECULE HAC: min={min(hv)} max={HAC_MAX} mean={np.mean(hv):.4f} "
  f"median={np.median(hv)} n={len(hv)}")
out["db_unique_molecules"] = int(len(uniq_mol))
out["db_hac_min"] = int(min(hv))
out["db_hac_max"] = int(HAC_MAX)
out["db_hac_mean_unique_molecules"] = float(np.mean(hv))
out["db_hac_median_unique_molecules"] = float(np.median(hv))

# ------------------------------------------------- DB element sets, 4 units
el_mol = Counter()
for m in uniq_mol:
    for e in (elset(m) or ()):
        el_mol[e] += 1
out["db_elements_unique_molecules"] = dict(sorted(el_mol.items()))
p(f"elements over unique molecules = {dict(sorted(el_mol.items()))}")

pair_key = list(zip(db_mol, db_f1, db_f2))
pair_df = pd.DataFrame(pair_key, columns=["m", "f1", "f2"]).drop_duplicates()
uniq_pair_mol = pd.unique(pair_df["m"])
p(f"db unique (molecule,f1,f2) pairs = {len(pair_df)}; their unique parents = "
  f"{len(uniq_pair_mol)}")
el_pair = Counter()
for m in uniq_pair_mol:
    for e in (elset(m) or ()):
        el_pair[e] += 1
out["db_unique_pairs"] = int(len(pair_df))
out["db_unique_pair_parents"] = int(len(uniq_pair_mol))
out["db_elements_unique_pairs"] = dict(sorted(el_pair.items()))
pHAC = [hac(m) for m in uniq_pair_mol]
out["db_unique_pair_hac_max"] = int(max(pHAC))
out["db_unique_pair_hac_mean"] = float(np.mean(pHAC))
p(f"elements over unique pairs = {dict(sorted(el_pair.items()))}; "
  f"pair-parent HAC max = {max(pHAC)}")

el_row = Counter()
for m in db_mol:
    for e in (elset(m) or ()):
        el_row[e] += 1
out["db_elements_row_weighted_parent"] = dict(sorted(el_row.items()))
p(f"elements row-weighted over raw DB rows (parent molecule) = "
  f"{dict(sorted(el_row.items()))}")

# row-weighted over BOTH fragment columns of every raw row
row_frag_el = Counter()
for f in db_f1:
    for e in (elset(f) or ()):
        row_frag_el[e] += 1
for f in db_f2:
    for e in (elset(f) or ()):
        row_frag_el[e] += 1
out["db_elements_row_weighted_fragments"] = dict(sorted(row_frag_el.items()))
p(f"elements row-weighted over raw DB row FRAGMENTS = "
  f"{dict(sorted(row_frag_el.items()))}")

uniq_frag = pd.unique(pd.concat([db_f1, db_f2]))
el_ufrag = Counter()
for f in uniq_frag:
    for e in (elset(f) or ()):
        el_ufrag[e] += 1
out["db_unique_fragments"] = int(len(uniq_frag))
out["db_elements_unique_fragments"] = dict(sorted(el_ufrag.items()))
p(f"elements over {len(uniq_frag)} unique fragments = {dict(sorted(el_ufrag.items()))}")

out["db_element_set_union_all_units"] = sorted(
    set(el_mol) | set(el_pair) | set(el_row) | set(el_ufrag) | set(row_frag_el))

# ------------------------------------------------- B) exact-row join
db_triples = set(pair_key)
db_triples_swapped = set(zip(db_mol, db_f2, db_f1))
p(f"db duplicate raw rows for same (molecule,f1,f2) = {len(db) - len(db_triples)}")

b_keys = list(zip(b_mol, b_f1, b_f2))
exact = [k in db_triples for k in b_keys]
exact_swap = [(not e) and (k in db_triples)
              for e, k in zip(exact, zip(b_mol, b_f2, b_f1))]
exact_any = [e or s for e, s in zip(exact, exact_swap)]
p(f"BENCHMARK exact (molecule,fragment1,fragment2) = {sum(exact)}/{len(b_keys)}")
p(f"  extra if fragment order swapped = {sum(exact_swap)} "
  f"(any order total {sum(exact_any)})")
out["benchmark_exact_rows_stated_order"] = int(sum(exact))
out["benchmark_exact_rows_any_order"] = int(sum(exact_any))
out["benchmark_exact_rows_swapped_only"] = int(sum(exact_swap))

db_fp = set(zip(db_f1, db_f2)) | set(zip(db_f2, db_f1))
out["benchmark_fragments_match_any_parent"] = int(
    sum(k in db_fp for k in zip(b_f1, b_f2)))

# ------------------------------------------------- C) parent molecule present
db_mol_set = set(uniq_mol)
pp = b_mol.isin(db_mol_set)
p(f"BENCHMARK parent molecule present in db = {int(pp.sum())}/{len(bench)}")
out["benchmark_parent_present"] = int(pp.sum())

# ------------------------------------------------- D) HAC beyond db max
b_hac_map = {m: hac(m) for m in pd.unique(b_mol)}
bh = b_mol.map(b_hac_map)
p(f"benchmark unique molecules={b_mol.nunique()} HAC min={bh.min()} max={bh.max()} "
  f"mean={bh.mean():.4f} median={bh.median()} sd={bh.std(ddof=1):.4f} "
  f"unparsable={int(bh.isna().sum())}")
out.update(
    benchmark_unique_molecules=int(b_mol.nunique()),
    benchmark_hac_min=int(bh.min()),
    benchmark_hac_max=int(bh.max()),
    benchmark_hac_mean=float(bh.mean()),
    benchmark_hac_median=float(bh.median()),
    benchmark_hac_sd=float(bh.std(ddof=1)),
)

beyond = bh > HAC_MAX
hok = bh <= HAC_MAX
p(f"cutoff = db HAC max = {HAC_MAX}; BENCHMARK HAC > {HAC_MAX} = {int(beyond.sum())} "
  f"({100*beyond.mean():.4f}%)")
out["benchmark_beyond_db_hac_max"] = int(beyond.sum())
out["benchmark_beyond_db_hac_max_pct"] = float(100 * beyond.mean())
out["benchmark_at_or_below_db_hac_max"] = int(hok.sum())
for c in range(5, 15):
    out[f"benchmark_hac_gt_{c}"] = int((bh > c).sum())
p("  [sens] benchmark HAC > c: " +
  ", ".join(f"{c}:{int((bh>c).sum())}" for c in range(6, 15)))

# ------------------------------------------------- E) decomposition
ex = pd.Series(exact, index=bench.index)
pps = pd.Series(pp.values, index=bench.index)
hoks = pd.Series(hok.values, index=bench.index)
p(f"inside by exact-row={int(ex.sum())}  by parent={int(pps.sum())}  "
  f"by HAC<={HAC_MAX}={int(hoks.sum())}")
p(f"exact & parent={int((ex & pps).sum())}  exact & HAC-ok={int((ex & hoks).sum())}  "
  f"parent & HAC-ok={int((pps & hoks).sum())}  exact without parent={int((ex & ~pps).sum())}")
out.update(
    exact_and_parent=int((ex & pps).sum()),
    exact_and_hac=int((ex & hoks).sum()),
    parent_and_hac=int((pps & hoks).sum()),
    exact_without_parent=int((ex & ~pps).sum()),
    outside_by_parent_or_hac=int((~(pps & hoks)).sum()),
    outside_by_parent=int((~pps).sum()),
    outside_by_hac=int((~hoks).sum()),
)
big = bench.loc[beyond.values]
bkey_big = list(zip(big["parent_structure_smiles"], big["fragment_a_smiles"],
                    big["fragment_b_smiles"]))
out["beyond_hac_parent_present"] = int(
    big["parent_structure_smiles"].isin(db_mol_set).sum())
out["beyond_hac_exact_rows"] = int(sum(k in db_triples for k in bkey_big))
p(f"of the {int(beyond.sum())} HAC>{HAC_MAX} rows: parent present="
  f"{out['beyond_hac_parent_present']}, exact row present={out['beyond_hac_exact_rows']}")

# ------------------------------------------------- F) benchmark elements
bel = Counter()
for m in pd.unique(b_mol):
    for e in (elset(m) or ()):
        bel[e] += 1
bfel = Counter()
for f in pd.unique(pd.concat([b_f1, b_f2])):
    for e in (elset(f) or ()):
        bfel[e] += 1
out["benchmark_parent_elements"] = dict(sorted(bel.items()))
out["benchmark_fragment_elements"] = dict(sorted(bfel.items()))
p(f"benchmark parent elements = {dict(sorted(bel.items()))}")
p(f"benchmark fragment elements = {dict(sorted(bfel.items()))}")
out["beyond_hac_values"] = sorted(int(v) for v in set(bh[beyond].tolist()))
out["benchmark_hac_gt10_examples"] = {
    k: int(v) for k, v in bh[beyond].sort_values(ascending=False).head(8).items()
}

res = {
    "generated_by": "output/analysis/verify_2_coverage.py (independent recomputation)",
    "claim": "HAC max 10; elements {C,N,O}; 331 exact rows; 369 parent present; 150 beyond HAC 10",
    "recomputed": out,
    "log": report,
}
(Path(__file__).with_suffix(".json")).write_text(json.dumps(res, indent=2, default=str))
p("wrote " + str(Path(__file__).with_suffix(".json")))