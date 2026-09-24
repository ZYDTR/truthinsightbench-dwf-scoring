# Data and Method Provenance Log

Task ID: `EarthScience_01_glacial_fluvial_erosion`
T0 (evidence boundary): `2024-10-17`

## 1. Scientific inputs used

All core claims in `output/Result.md` rest exclusively on the frozen workspace payload under
`data/`. No external scientific dataset was downloaded, substituted, or merged into the
evidentiary analysis. No network access was used at any point in this run.

| # | Path | Role in this analysis | Bytes | sha256 (verified against `data_manifest.json`) |
|---|------|----------------------|-------|-----------------------------------------------|
| 1 | `data/raw/glacial_erosion_Earth.tsv` | Primary table, glacial erosion-rate estimates (427 rows, 24 columns) | 117004 | `b49192d77082fd431963ca67278d9face1079e8054711b84296554573c24e2cf` |
| 2 | `data/raw/nonglacial_erosion_Earth.tsv` | Primary table, nonglacial erosion-rate estimates (6288 rows, 26 columns) | 1662513 | `214063142d936c3a5408583b682397042392b4d46e0443e30c88811fbe9fee89` |
| 3 | `data/raw/prior compilations/OCTOPUS/octopusdata.tsv` | Raw measurement columns behind the nonglacial cosmogenic-detrital stratum (3342 rows, 69 columns) | 1600386 | `dd7620a0078af24a6e1882faaa76c22bc74d7d0243d265e5bf2d1fb6174ee4c0` |
| 4 | `data/raw/prior compilations/OCTOPUS/octopusages.tsv` | Derived exposure ages paired row-for-row with `octopusdata.tsv` (3342 rows) | 66818 | `89a9feb971dac5dbe850cf786083ecd0bfe5a10f9ea877ad39f9b9d87e241665` |
| 5 | `data/raw/prior compilations/OCTOPUS/octopusexposureage.jl` | The published derivation script; documents how exposure age was computed from the raw nuclide columns. Method provenance only, not evidence. | 2397 | `17172ed350479323955a2c3c13d6531286e3232df389b7bf2badceef205230b8` |
| 6 | `data/raw/prior compilations/CHELSEA/*.tsv`, `annualprecipitation.jl`, `references.txt` | Basin-mean precipitation lookups extracted from CHELSA climatologies; used only for the precipitation covariate. | — | as listed in `data_manifest.json` |
| 7 | `data/raw/prior compilations/GloRiSe/GloRiSe combined.tsv` | Sediment-yield compilation contributing the GloRiSe subset (1078 rows, 26 columns) | 253975 | `a419086603c8a6a52742cee08abf9a953958fe921d4109f2759e3e22ef5fbff1` |
| 8 | `data/raw/prior compilations/Chen et al. 2023/Chen_et_al_2021_short-term_erosion_rate_data.csv` | Short-term erosion-rate compilation contributing the Chen subset (1521 rows, 11 columns) | 153924 | `5a0668568db63339e8ed83c2196edc258b2c0d975c120588fca90503842fcb4f` |
| 9 | `data/raw/slope_glacial.tsv`, `data/raw/slope_nonglacial.tsv` | Headerless per-location slope values (461 and 6271 lines) | — | as listed in `data_manifest.json` |
| 10 | `data/raw/lithology_parsed.tsv`, `lithology_parsed_nonglacial.tsv`, `lithology_unparsed*.tsv`, `lithology_reference*.tsv` | Lithology lookup tables keyed by coordinate hash | — | as listed in `data_manifest.json` |
| 11 | `data/raw/beryllium.tsv` | Chemical analyte table (70 rows, dissolved/colloidal Be in estuaries and groundwater). Not erosion-rate data; excluded. | 9363 | `cde5b0794baf9622e74e9749fd38514cfd9ab63fb54250baf6a15777c983e4d2` |

Byte counts and sha256 values were recomputed locally from disk and compared field-by-field
against `data_manifest.json`. See `output/analysis/w5_audit/` for the per-file receipt.

## 2. External sources

None were used. The analysis is self-contained within the frozen payload.

No external literature search was performed, so no source in this workspace is subject to a
pre-`T0` first-public test. Background concepts applied below (log-space regression of
denudation rates against observation interval; spallation production-depth relations in
terrestrial cosmogenic nuclide dating) are standard textbook relations that were public long
before `T0 = 2024-10-17`, and they are applied here only as methods, never as evidence for a
workspace claim. The one formal relation used as an interpretive frame is stated and derived
inside `output/Result.md` from the workspace's own columns, so that a reader can verify it
without consulting any external text:

- For a nuclide concentration `N` produced at rate `P` in rock of density `rho`, equilibrating
  over a spallation attenuation length `Lambda`, the steady-state surface-erosion rate is
  `E = P * Lambda / (rho * N)`. The product `E * t` with exposure age `t = N / P` is therefore
  `Lambda / rho`, a material constant independent of the site. This is the identity tested
  empirically in Finding 1.

## 3. Environment

| Item | Value |
|------|-------|
| Python | 3.11.2 |
| pandas | 3.0.5 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| statsmodels | 0.15.0 |
| matplotlib | 3.11.2 |

`scipy`, `statsmodels` and `matplotlib` were not present at session start and were installed
from the environment's configured package index. All other libraries were preinstalled.

## 4. Integrity statement

- Every number in `output/Result.md` was produced by executing the scripts in `output/analysis/`.
- No command output, statistic, or figure was hand-written or estimated.
- Failed and superseded attempts are recorded in `output/analysis/REPRODUCE.md` rather than discarded.
- The workspace `data/` directory was treated as the complete and only task dataset.
