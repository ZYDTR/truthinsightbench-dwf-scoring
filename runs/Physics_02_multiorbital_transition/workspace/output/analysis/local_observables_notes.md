# local_observables — notes

Task: total magnetic moment and charge fluctuations vs control ratio U/W (W=1).

## Method
- Loaded the normalized view TSV (49 rows, U/W = 0..8).
- Monotonicity: successive differences; moment should be >= 0 (increasing),
  charge should be <= 0 (decreasing). Violations counted.
- Regression: numpy.polyfit degree 1; R^2 from (1 - SSres/SStot).
- Cross-check: extracted table_007.dat from the deidentified zip and parsed
  its real column layout.

## Data provenance
- WS/data/normalized_views/control_vs_local_observables.tsv
- WS/data/source_data/all_source_data_deidentified.zip :: table_007.dat

## Key numbers
| quantity | value |
|---|---|
| rows (U/W) | 49 |
| moment @ U/W=0 | 0.750015 |
| moment @ U/W=8 | 1.9703 |
| charge @ U/W=0 | 0.99998 |
| charge @ U/W=8 | 0.0233708 |
| moment min successive diff | 0.004089999999999927 |
| moment violations (decreases) | 0 |
| charge min successive diff | -0.047308999999999934 |
| charge violations (increases) | 0 |
| moment slope | 0.16860554510571155 |
| moment intercept | 1.0828422662458388 |
| moment R^2 | 0.7184900505589695 |
| charge slope | -0.12154595894115615 |
| charge intercept | 0.660211230208829 |
| charge R^2 | 0.6985445829837197 |
| raw table_007 real column count | 2 |
| max rel diff moment (TSV vs raw) | 0.0 |
| raw charge available | True |

## Raw-table cross-check (table_007.dat)
The header comment mentions BOTH total magnetic moment and charge fluctuations,
which could suggest a 3-column table. In reality each data row has only
2 column(s); the file holds TWO stacked 2-column blocks:
  block 1: U/W, total magnetic moment T^2
  block 2: U/W, charge fluctuations delta n
Charge fluctuations ARE present in the raw table (block 2), so
raw_check_charge_available = True.
The TSV moment values match the raw moment block on the shared U/W set with
max relative difference 0.0.

## Caveats
- U/W grid is not uniform: steps of 0.1 from 0 to 4, then steps of 0.5 from
  4 to 8 (49 points total).
- R^2 is for a single linear fit over the whole range; the moment curve is
  concave (saturating), so a straight line is only a global approximation.
- Max relative difference is computed as |TSV - raw| / |raw| over shared U/W.
