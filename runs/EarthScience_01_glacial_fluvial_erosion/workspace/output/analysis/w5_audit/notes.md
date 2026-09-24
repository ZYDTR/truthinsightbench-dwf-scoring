# w5_audit — data inventory, integrity, missingness and independence audit

Script: `w5_audit.py` (run with `cd <PROJECT> && python3 output/analysis/w5_audit/w5_audit.py`).
All numbers below come from that execution; `results.json` holds the machine-readable version.
Evidence boundary: only `data/` under this project. No network, no external data.

## 1. File inventory and manifest integrity — CLEAN

21 files on disk, 21 entries in `data_manifest.json`, all under the `raw/` prefix.
Every file matches its manifest `bytes` **and** `sha256`. Zero mismatches, zero files present
on disk but absent from the manifest, zero manifest entries absent from disk.
There are no manifest entries with a path outside `raw/`, so the 21/21 correspondence is exact.

| file | bytes | sha256 (first 16) | status |
|---|---|---|---|
| raw/beryllium.tsv | 9363 | cde5b0794baf9622 | MATCH |
| raw/glacial_erosion_Earth.tsv | 117004 | b49192d77082fd43 | MATCH |
| raw/nonglacial_erosion_Earth.tsv | 1662513 | 214063142d936c3a | MATCH |
| (18 more) | — | — | all MATCH |

## 2. Table shapes

| file | rows | cols | format |
|---|---|---|---|
| raw/glacial_erosion_Earth.tsv | 427 | 24 | TSV, header |
| raw/nonglacial_erosion_Earth.tsv | 6288 | 26 | TSV, header |
| raw/slope_glacial.tsv | 461 | 1 | TSV, **headerless single column** |
| raw/slope_nonglacial.tsv | 6271 | 1 | TSV, **headerless single column** |
| raw/lithology_parsed.tsv | 229 | 7 | TSV, header |
| raw/lithology_parsed_nonglacial.tsv | 4900 | 7 | TSV, header |
| raw/lithology_unparsed.tsv | 229 | 11 | TSV, header |
| raw/lithology_unparsed_nonglacial.tsv | 4900 | 11 | TSV, header |
| raw/beryllium.tsv | 70 | 16 | TSV, header |
| raw/lithology_reference{,_nonglacial}.tsv | 1 | 4 | TSV, headerless citation |
| CHELSEA/precipitation_glacial.tsv | 444 | 3 | TSV, header |
| CHELSEA/precipitation_nonglacial.tsv | 6271 | 3 | TSV, header |
| CHELSEA/references.txt | — | — | prose |
| CHELSEA/annualprecipitation.jl | — | — | Julia source |
| Chen et al. 2023/Chen_...data.csv | 1521 | 11 (+2 BOM artefacts) | CSV, header, **latin-1 encoding** |
| GloRiSe/GloRiSe combined.tsv | 1078 | 26 | TSV, header |
| OCTOPUS/octopusdata.tsv | 3342 | 71 | TSV, header |
| OCTOPUS/octopusages.tsv | 3342 | 2 | TSV, header |
| OCTOPUS/octopusexposureage.jl | — | — | Julia source |
| raw/source_notes.txt | — | — | 1 line of placeholder text |

Note on the Chen CSV: it is not UTF-8. `pd.read_csv(..., encoding='latin-1')` is required;
the Koppen–Geiger column header contains a `0x92` byte that decodes as `¡V` under latin-1.

## 3. Row-level audit (trails in `audit_*.csv`)

Glacial (n=427):
- Erosion rate: 0 missing, **0 rows ≤ 0**. Range 4.0e-4 – 1000 mm/yr (median 0.55).
- Time interval: 0 missing, 0 ≤ 0. Range 0.0192 – 2.5e8 yr (median 1270).
- Area: 35 missing; **100 rows have Area = 1e-8 km² exactly** (a placeholder, not a measurement).
  All 100 are `Balter-Kennedy et al., 2021` (24) or `Marrero et al., 2018`-style rows; see CSV.
  Non-placeholder range 13.35 median, 2.17e6 max. No row above 1e7.
- Latitude: **2 rows missing** (`Grosval'd and Glazovskiy, 2008`, Locality "Earth" — global
  estimates, no coordinates). Longitude also missing on those 2. No out-of-range latitudes.
- Precipitation: 2 missing (same two rows). Range 29.2 – 7857.1 mm/yr.
- Slope: 2 missing (same two rows), none ≤ 0.
- Exact full-row duplicates: **0**. Duplicates on the analysis key: 7 rows.
- Time interval > 1e8 yr: 1 row (long-term / geologic-age estimate).

Nonglacial (n=6288):
- Erosion rate: 0 missing, **4 rows exactly 0.0** (all `Marrero et al., 2018`,
  Cosmogenic surface). log-rate models must drop these 4.
- Time interval: **160 missing**, all `Marrero et al., 2018`; 0 rows ≤ 0.
  Range 0.00274 – 2.0e7 yr (median 1201.2).
- Area: 68 missing; **224 rows have Area = 1e-8 km² exactly** (194 `Marrero et al., 2018`,
  30 `This study`). No row above 1e7.
- Latitude/Longitude: 0 missing, 0 out of range. Precipitation: 0 missing.
- Exact full-row duplicates: **0**. Duplicates on the analysis key: 198 rows.
- Time interval > 1e8 yr: 0 rows.

The two Area = 1e-8 placeholders and the 160 missing intervals are the two biggest
model-sample hazards; both are concentrated in `Marrero et al., 2018`.

Important counting nuance found while auditing: `Area = 1e-8 km²` is **not** the same as
"missing". The column holds a real, finite, strictly positive number for those 224 + 100 rows,
so `log10(Area)` is finite (−8) and any `dropna()` keeps them. They will silently distort any
Area covariate unless explicitly excluded. Use a threshold, not `notna()`, to drop them.

## 4. Exhumation unit identity — VERIFIED but weaker than stated

`Exhumation (m)` is populated for **81 glacial and 11 nonglacial rows** (matches the briefing).
Testing `Erosion rate × Time interval == Exhumation × 1000`:

- Glacial: only **11 of the 81** rows satisfy the identity to <1e-9 relative; 20 to <1e-6.
  48 of 81 have ratio (ER×TI)/(Exhumation×1000) in [0.99, 1.01].
  The identity holds tightly **per reference**, not per row:
  `Balter-Kennedy et al., 2021` (24 rows, ratio 0.925–1.064), `Steinemann et al., 2020`
  (13 rows, ratio 0.56–7.61) and `Young et al., 2016` (8 rows, ~1.00) dominate the near-1 group
  but carry internal scatter, while single-row references like `Bell and Laine, 1985`,
  `Corbett et al., 2019`, `Dühnforth et al., 2010`, `Kaszycki and Shilts, 1980`,
  `Medvedev et al., 2008`, `Rocchi et al., 2006`, `Stern et al., 2005`, `White, 1972`
  hit the identity exactly. Extreme ratios (800, 1028, 1875) belong to
  `Fitzgerald et al., 1995`, `Zeitler et al., 2001`, `Foster et al., 1994` — those
  "Exhumation (m)" values are ≤ 0.05 m and are clearly a different quantity.
- Nonglacial: **0 of 11** rows satisfy the identity. Ratio (ER×TI)/(Exhumation×1000) is
  333 for `Burbank et al., 2000` rows (ER=3, TI=2e7 → 6e7 mm = 6e4 m, but Exhumation = 6 m).

**Conclusion: the mm-vs-m unit identity is NOT a general property of the table.** It is a
coincidence of a subset of glacial rows (mostly `This study` compilation) whose Exhumation was
back-computed as rate × interval. Treat `Exhumation (m)` as its own measured column, and do not
use it as a check on `Erosion rate × Time interval`.

## 5. Independence structure

| | glacial | nonglacial | pooled |
|---|---|---|---|
| rows | 427 | 6288 | 6715 |
| unique Reference | 154 | 265 | 417 |
| unique Compilation | 13 | 6 | 17 |
| unique Locality | 122 | 1819 | 1937 |
| unique (lat,lon) pair | 229 | 4900 | 5116 |
| unique Reference DOI | 120 | 251 | 369 |
| max rows from one Reference | 32 (`Berger and Spotila, 2008`) | 512 (`Milliman and Farnsworth 2013`) | 512 |
| share of that Reference | 7.5% | 8.1% | 7.6% |
| rows per Reference (median) | 1 | 13 | 7 |

Top compilations: glacial `This study` 294 (68.9%), `Hallet et al., 1996` 62 (14.5%),
`Balter-Kennedy et al., 2021` 32 (7.5%). Nonglacial `OCTOPUS` 3457 (55.0%),
`Chen et al., 2022` 1521 (24.2%), `GloRiSe` 1057 (16.8%), `Marrero et al., 2018` 194 (3.1%).
93.9% of nonglacial rows come from just three prior compilations — effective n is far below 6288.

Only **2 References are shared across both tables** (`Glotzbach et al., 2013`,
`Willenbring Staiger et al., 2006`), so glacial/nonglacial contrasts are mostly between
disjoint literature sets, not paired comparisons.

**OCTOPUS row expansion (important for independence):** `octopusdata.tsv` has 3342 rows,
one per sample, with 236 rows carrying an Al-26 rate and 3221 carrying a Be-10 rate.
236 + 3221 = **3457 = the nonglacial OCTOPUS row count exactly**. The main table therefore
emits one row per (sample, nuclide) pair; the two rows for a sample are not independent.

## 6. Cross-file consistency: companion vs per-location

Key used throughout: `(Latitude rounded 5 dp, Longitude rounded 5 dp)`, with a single
`MISSING|MISSING` group for the 2 coordinate-less glacial rows (`.nunique()` silently drops NA
and would undercount by 1 — this is why some naive counts read 228 instead of 229).

| file | rows | unique keys | per-row or per-location |
|---|---|---|---|
| slope_glacial.tsv | 461 | (no coords) | per-row companion of the **nonglacial** table, then of glacial (see below) |
| slope_nonglacial.tsv | 6271 | (no coords) | per-row companion of nonglacial **minus 17 rows** |
| precipitation_glacial.tsv | 444 | 232 | per-row companion + 4 extra Antarctic keys |
| precipitation_nonglacial.tsv | 6271 | 4896 | per-row companion, exactly mirroring main rows-per-key |
| lithology_parsed.tsv | 229 | 229 | **per-unique-location** (1 row per key) |
| lithology_parsed_nonglacial.tsv | 4900 | 4900 | **per-unique-location** (1 row per key) |

Evidence details:

- **precipitation_nonglacial.tsv**: for every key present in both files the aux row count equals
  the main row count (max 154 rows/key on both sides); values agree to 1e-9 on 100% of the
  51157 inner-join rows (r = 1.0). It is a strict per-row companion. 4 keys
  (`-77.9|160.92`, `-77.54999|160.0897`, `-77.85|160.92`, `-76.98366|161.52648`) appear only in
  the main table — they are the appended Antarctic rows.
- **precipitation_glacial.tsv**: same per-row relationship for the shared keys
  (values agree on 99.9% of 3193 joined rows), **plus** 4 extra Antarctic keys carrying 17 extra
  rows. Those 17 rows are exactly main-table glacial rows 410–426 in the same order, so the file
  is the glacial main table with 17 duplicate Antarctic rows inserted mid-file.
- **lithology_parsed*.tsv**: exactly one row per key (`max_rows_per_key = 1`), and the key
  sequence equals the main table's **first-occurrence order**
  (`lithology_parsed_nonglacial` matches 4900/4900; `lithology_parsed` matches 229/229). These
  are per-unique-location lookups, not per-row companions.
- **slope_nonglacial.tsv**: dropping main rows 6241–6247 (Brook et al., 1995) and 6265–6274
  (Margerison et al., 2005) makes the file align positionally with 100% agreement after 5-dp
  rounding (median relative difference 2.6e-8). The 132 rows with relative difference >1e-6 are
  entirely the 5-decimal display rounding of the main table.
- **slope_glacial.tsv**: has 461 rows for 427 main rows. difflib alignment of the rounded slope
  sequence finds three insert blocks — file[202:209], file[295:316], file[353:363] — and one
  main block (288:292). Excising them reconciles 423 vs 423 rows, with 99.53% exact 5-dp
  agreement and the 12 residual differences all explained by display rounding. The insert
  blocks contain (a) the 7 Brook et al., 1995 and (b) the 10 Margerison et al., 2005 slope values
  that now live in the nonglacial table, and (c) a 21-row `Hasholt and Hagedorn, 2000`
  Zackenbergelven block whose main-table counterparts are 4 `This study` rows with slope 64.31303.
  Four main rows (64.313) are absent from the slope file.

**The 17 "Antarctic" rows are the single root cause of every row-count discrepancy.**
They are present in the nonglacial main table but absent from all four nonglacial companion
files (slope, precipitation, lithology parsed, lithology unparsed). They are the
`Brook et al., 1995` + `Margerison et al., 2005` samples of Arena Valley, Beardmore Glacier and
Mackay Glacier (Antarctica, latitudes -76.98 to -77.90). They also appear duplicated inside
`precipitation_glacial.tsv`. Conclusion: the companion files were generated **before** those
rows were appended to `nonglacial_erosion_Earth.tsv`, and the append borrowed both the
locations and the (glacial) precipitation values.

## 7. Unit audit

### Chen et al. 2023 CSV
Column `Short-term erosion rate (mm kyr^-1)`. The header states mm per kyr. The main table's
`Chen et al., 2022` subset (1521 rows, all `Volumetric`) stores mm/yr.

Comparison (1477 shared location strings, aggregated per location):
- Pearson r = 1.0000 (log-space r = 1.0000)
- median ratio main/csv = **0.001** exactly; 77.1% of locations hit exactly 0.001;
  **100% lie within 1% of 0.001** (5th–95th pct 0.00099999–0.00100000)

Conclusion: the CSV column is mm per 1000 yr; the main table divides by 1000 to store mm/yr.
The residual scatter is only because the main table's mm/yr value is rounded to fewer decimals
(e.g. 13.5048 vs 13504.7537/1000 = 13.504754). Row-order is not preserved for the last ~510
rows, so **align on `Location`+`Reference`, never on row position** (the CSV is sorted
differently after the USGS block). The `Basin area (km^2)` column agrees positionally on
only 62.5% of rows — another reason not to trust positional alignment.

### GloRiSe combined.tsv (1078) vs nonglacial GloRiSe subset (1057)
Difference is exactly **21 rows, fully accounted for**:
- All are `Hasholt and Hagedorn, 2000`, basin `Zackenbergelven` (Greenland).
  Sample IDs `GRL-ZKB-111111-HSH4` … `GRL-ZKB-111127-HSH1`, e.g. one per sample.
- 0 sample IDs in the main table are missing from the combined file, and
  `combined_rows - main_rows = sample_ids_in_combined_not_main = 21`.
- 4 of the 21 also lack a Longitude in the combined file.
- Separately, the two files spell six references differently (`Negrel and Grosbois 1999`
  → `Negrel and Grosbois, 1999`; `GEMS-GLORI (Meybeck and Ragu, 1998)` →
  `Meybeck and Ragu, 1998`; `Canfield 1997` → `Canfield, 1997`; `Chakrapani 2005` →
  `Chakrapani, 2005`; `Vale 1986` → `Vale, 1986`).
- After name normalisation the per-reference tally still shows `Canfield` 23 (main) vs 22
  (combined). This is **not** an extra main row: sample `USA-COL-111111-CAN1` (Colorado basin,
  6.4e5 km²) exists in the combined file **with a missing Reference** and the main table assigns
  it to `Canfield, 1997`. That NaN also explains why the combined file has 1078 rows but only
  1077 non-null Reference values. Row total reconciles exactly: 1078 − 21 = 1057.

**Unresolved and important:** the *values* of `Erosion rate (mm/yr)` in the two files disagree.
- In `GloRiSe combined.tsv` the column is exactly `TSS_km3_yr / Area_km2 × 1e6` (verified to
  1.3e-11 relative on all 1078 rows), i.e. an implicit 2700 kg/m³ sediment density
  (the implied density from `TSS_kg_s` is exactly 2700.0).
- The nonglacial main-table values are **not** that series: the ratio
  main/(combined) has median 2.16, IQR 1.16–8.38, max 472. A 125-combination brute-force search
  over all available columns (`TSS_km3_yr`, `TSS_kg_s`, `TSS_mg_L`, `Discharge_m3_s`,
  `Annual_Discharge_m3_s`, `Area`, `Sand/Silt/Clay_perc`, `filtersize_mum`) with exponents
  −2…+2 found no exact identity. The ratio is strongly anti-correlated with `TSS_mg_L`
  (r = −0.70 in log space) and varies with `Season` (median ratio: Spring 5.19, Autumn 3.49,
  Summer 2.01, Winter 1.15), so the main-table rate looks like a seasonally/differently
  aggregated recomputation rather than a relabelling.
- Same-unit (mm/yr) but a different numeric series. `Sediment flux (km3/yr)` in the main table
  **does** equal `TSS_km3_yr` (ratio 1.000000 ± 3e-6), so only the rate column was recomputed.

Domain workstreams must not treat the nonglacial `GloRiSe` stratum as a byte-for-byte copy of
`GloRiSe combined.tsv`.

### OCTOPUS
`octopusdata.tsv` 3342 rows; 236 with Al-26 rate + 3221 with Be-10 rate = 3457 = the nonglacial
OCTOPUS row count. The main table is an (sample × nuclide) expansion, and `Sample ID` is
duplicated on 397 nonglacial OCTOPUS rows. `octopusages.tsv` has the same 3342 rows and holds
only the two exposure-age columns.

## 8. Files NOT part of the erosion-rate evidence chain

| file | reason (column evidence) |
|---|---|
| raw/beryllium.tsv | 70 rows, columns `Be (nmol/kg)`, `Be dissolved`, `Be colloidal`, `DOC (nmol/kg)`, `Alkalinity (mM HCO3)`, `Sal (PSU)`, `T (C)`, `pH`, `Depth (m)` — estuarine beryllium chemistry. It shares the identifier columns `Reference`, `Reference DOI`, `Locality`, `Region`, `Type` with the main tables, which makes it easy to mistake for erosion data, but it has **no** Erosion rate / Time interval / Area / Slope / Precipitation column. |
| raw/lithology_reference.tsv, raw/lithology_reference_nonglacial.tsv | single 4-field citation for the Chorlton 2007 world geology map (identical to each other). Auxiliary provenance, not measurements. |
| prior compilations/CHELSEA/references.txt | prose bibliography for the CHELSA climatology product. |
| prior compilations/CHELSEA/annualprecipitation.jl | Julia code that generated the precipitation companions. |
| prior compilations/OCTOPUS/octopusexposureage.jl | Julia code that generated octopusages.tsv via CAIRN scalings. |
| raw/source_notes.txt | one line of placeholder text ("A description of all the files in this folder..."). |

## 9. Practical guidance for the other workstreams

1. Use `nonglacial` n = 6288 but drop the 4 zero-rate rows and the 160 missing intervals before
   any log fit, and state whether the 224 Area = 1e-8 placeholder rows are excluded.
2. Never quote nonglacial n as an effective sample size: 93.9% of rows come from 3 compilations,
   and the largest single Reference contributes 512 rows (8.1%).
3. For OCTOPUS, a per-sample aggregation is needed to remove the (sample × nuclide) expansion.
4. `Exhumation (m)` is not a general rate × time product — do not use it as an internal check.
5. Auxiliary files are keyed on (lat,lon) with the rules in section 6; lithology is
   per-unique-location (must be joined, never row-pasted), precipitation/slope are per-row
   companions valid only after accounting for the 17 Antarctic rows.
6. The two main tables share only 2 References, so cross-process contrasts are between
   largely disjoint literature, not paired sites.

## 10. What failed / is unresolved

- The exact construction of the nonglacial `GloRiSe` `Erosion rate (mm/yr)` values was not
  determined. The values are definitively not the bundled `GloRiSe combined.tsv` column of the
  same name, and no combination of the bundled columns reproduces them exactly.
- `slope_glacial.tsv` cannot be row-aligned by any simple rule (position, sorted latitude,
  sorted (lat,lon)); it required explicit block excision (file[202:209], file[295:316],
  file[353:363] vs main[288:292]). Its provenance is a concatenation of an older glacial slope
  column and values that later migrated to the nonglacial table.
- Note for the record: this directory also contains `audit_data.py` and `table_*.csv` files
  written by a concurrent process at 14:22, not by this workstream. They were left untouched;
  the w5_audit deliverable is `w5_audit.py`, `results.json`, `notes.md` and the
  `audit_*.csv` / `column_missingness_*.csv` / `companion_alignment.csv` /
  `independence_structure.csv` / `top_*.csv` / `chen_unit_check.csv` /
  `glorise_reconciliation.csv` / `non_erosion_files.csv` / `strata_sizes.csv` /
  `table_shapes.csv` / `file_inventory.csv` / `exhumation_identity_by_reference.csv` files.