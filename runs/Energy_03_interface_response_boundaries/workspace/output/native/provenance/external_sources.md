# External source log

Task: `Energy_03_interface_response_boundaries`. Boundary `T0 = 2022-01-01`.

**Correction notice.** The first version of this log attached wrong authors,
volumes and pages to several DOIs. An independent audit caught this and every
record below was re-resolved live through the Crossref REST API. The rule now
applied is stricter: **only fields actually returned by Crossref are printed.**
No author list, volume or page number is stated from memory. Where the report
cited a source that the log had mis-described, the source log — not the report
wording — has been corrected.

Rule applied: an external source is used **only** for background, methods, or
alternative explanations, and **only** if it was first public strictly before
`T0`. Every date below is the Crossref `issued` date-parts, recorded verbatim.
No external dataset was downloaded, substituted, or merged; the workspace
`data/` directory remains the complete direct evidence for every registered
finding.

No external search was used to identify or reconstruct the hidden source behind
the task. The literature queries were generic method/background queries.

| # | Use in this report | Title (as returned by Crossref) | Authors (as returned) | Journal, vol, pages | DOI | First public | Before T0? |
|---|---|---|---|---|---|---|---|
| S1 | Arrhenius rate form used as the conventional baseline (Findings 1, 2) | *Über die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker durch Säuren* | Arrhenius, S. | Z. Phys. Chem. **4U**, 226–248 | `10.1515/zpch-1889-0416` | 1889-07-01 | yes |
| S2 | "Light-off"/ignition–extinction framing for a saturating S-shaped activity curve (Findings 3, 4) | *Ignition and extinction in the catalytic oxidation of hydrocarbons over platinum* | Veser, G.; Schmidt, L. D. | AIChE J. **42**, 1077–1087 | `10.1002/aic.690420418` | 1996-04 | yes |
| S3 | Coverage-change mechanism during light-off; supports the caveat that an S-curve does not identify a unique surface mechanism (Finding 4 limitations) | *Shedding light on catalytic ignition: coverage changes during CO oxidation on Pd(110)* | Bowker, M.; Jones, I. Z.; Bennett, R. A.; et al. | Catal. Lett. **51**, 187–190 | `10.1023/a:1019013807493` | 1998-05 | yes |
| S4 | Structured/heterogeneous-surface origin of a non-constant *apparent* activation energy — the standard alternative explanation for the exponent spread in Finding 2 | *Apparent activation energies on catalysts with heterogeneous surfaces II. Dual-site catalysts and compensation effect* | Bagg, J. | J. Catal. **16**, 377–385 | `10.1016/0021-9517(70)90235-6` | 1970-03 | yes |
| S5 | Modern account of apparent activation energy in structure-sensitive heterogeneous catalysis; justifies calling the measured 7–11 kJ/mol "apparent" (Finding 2) | *On Apparent Activation Energy of Structure Sensitive Heterogeneous Catalytic Reactions* | Murzin, D. Yu. | Catal. Lett. **149**, 1455–1463 | `10.1007/s10562-019-02772-0` | 2019-04-03 | yes |
| S6 | Interface-catalysis context only; **not** used to explain any workspace result | *H₂ reduction of CeO₂(111) surfaces via boundary Rh–O mediation* | Xu, J.; Overbury, S. H. | J. Catal. **222**, 167–173 | `10.1016/j.jcat.2003.10.011` | 2004-02-15 | yes |

## Which sources are actually cited in the report

The first version of this log claimed "Only S1–S6 appear anywhere in the report".
That was wrong: `Result.md` cites **S2, S3, S4 and S5 only**. S1 and S6 are
retained in this log as background that was consulted but is **not** cited in
the report. This is stated so the log does not overstate the report's citation
base.

## Sources located but **not** used

`10.1021/cr020359o` (a putative review of CO oxidation on platinum) was checked
and returns no parsable record (HTTP 404 at the time of the query). It is
therefore **not** cited and carries no weight in this report.

## Boundary compliance statement

Every source above has an issued date strictly before `T0 = 2022-01-01`,
independently re-verified. External sources are used exclusively in background,
method-justification, or alternative-explanation context. No external numeric
value is compared against, fitted to, or merged with the workspace curves. All
quantitative results under `../results/` were computed in this run from
`data/anonymous_condition_response_curves.tsv` and
`data/independent_condition_check_curves.tsv` only.

**Honest limitation.** This log records titles, authors, journals and dates as
returned by a Crossref metadata query. That is bibliographic metadata, not a
full-text reading: no external full text was retrieved or read for this task,
and none of these sources is load-bearing for any registered finding.