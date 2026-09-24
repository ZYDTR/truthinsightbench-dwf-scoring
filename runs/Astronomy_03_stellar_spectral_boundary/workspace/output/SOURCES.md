# Background Reference Sources (pre-T0)

**Scope note.** Every source below is a background, methods, or candidate-alternative-explanation
reference only. T0 for this task is **2023-01-01**, and every source listed here was first public
**strictly before** that date. These references are **NOT direct evidence for any claim about the
bundled data** under `data/`. The workspace tables (`photon_flux_grid.tsv`,
`normalized_spectral_response.tsv`) are the complete and only dataset; no external dataset was
downloaded, merged, or substituted, and no external source was used to identify or reconstruct the
source article behind this task.

**Date-verification protocol used for all entries.** Dates were verified against primary
registration/publisher metadata, retrieved live:
(i) the Crossref REST API (`https://api.crossref.org/works/<DOI>`, fields `issued`,
`published-print`, `published-online`, `created`) — publisher-deposited metadata;
(ii) the arXiv API (`http://export.arxiv.org/api/query?id_list=<id>`, field `<published>`) and the
arXiv abstract page submission history (`https://arxiv.org/abs/<id>`, which prints the date of each
version) — for the first-public preprint date; and
(iii) OpenAlex (`https://api.openalex.org/works/doi:<DOI>`, field `publication_date`) as an
independent corroboration where available.
The "first public" date quoted is the **earliest** verifiable public appearance (arXiv v1 if the
paper was posted before journal publication, otherwise the publisher online/print date). All
retrievals were done on 2026-09-19 by plain HTTPS `curl`. All dates below are 1901–2017, i.e.
decades before T0.

---

## 1. Planck, M. 1901 — "Ueber das Gesetz der Energieverteilung im Normalspectrum"

- **Citation:** Planck, M. 1901, Annalen der Physik, 309(3), 553–563.
- **Stable identifier:** DOI `10.1002/andp.19013090310`
- **First public:** 1901 (issue 3 of vol. 309, January 1901).
- **How verified:** Crossref record for DOI `10.1002/andp.19013090310` returned
  `issued = 1901-01`, `published-print = 1901-01`, confirming 1901; OpenAlex record for the same DOI
  returned `publication_date = 1901-01-01`. Both are strictly before 2023-01-01 by more than a
  century.
- **What it supports:** The Planck law fixes the spectral distribution of radiation and, via
  E = hc/lambda, makes the conversion between photon flux and energy flux an explicit factor of
  wavelength; it therefore supports the candidate mechanism that any "peak" or "boundary" read off
  a photon-flux grid is a function of the plotted quantity (photon number vs energy), not an
  invariant property of the star.

## 2. Gustafsson, B., Edvardsson, B., Eriksson, K., Jørgensen, U. G., Nordlund, Å., & Plez, B. 2008 — "A grid of MARCS model atmospheres for late-type stars. I. Methods and general properties"

- **Citation:** Gustafsson, B., Edvardsson, B., Eriksson, K., Jørgensen, U. G., Nordlund, Å., &
  Plez, B. 2008, A&A, 486, 951–970.
- **Stable identifier:** DOI `10.1051/0004-6361:200809724` (arXiv:0805.0554; ADS bibcode
  `2008A&A...486..951G`)
- **First public:** 2008-05-05 (arXiv v1); journal online 2008-05-29, print 2008-08.
- **How verified:** arXiv API for `0805.0554` returned `<published>2008-05-05T13:56:38Z</published>`;
  the arXiv abs page (`https://arxiv.org/abs/0805.0554`) shows submission history "[v1] Mon, 5 May
  2008"; Crossref for the DOI returned `published-online = 2008-05-29`, `published-print = 2008-08`;
  OpenAlex returned `publication_date = 2008-05-29`. Earliest verifiable public date 2008-05-05,
  strictly before T0.
- **What it supports:** A standard, homogeneous grid of cool-star model atmospheres and fluxes
  computed over the same wavelength domain used here, and a canonical statement of the blanketing
  mechanism: the blocked radiation heats deeper layers ("back-warming", +10% in the outermost
  layers if line opacity is neglected), so that line blanketing both depresses the flux where the
  lines are and redistributes it to other wavelengths; supports the candidate mechanism that
  abundance/metallicity effects on emergent flux are strongly wavelength-dependent and do not keep
  one sign. It also documents that model SEDs are supplied on a **discrete wavelength grid**
  (about 108,000 points at sampling R = lambda/delta-lambda = 20,000), and that missing weak lines
  produced a known model flux excess "in particular in the ultraviolet and violet" — background for
  both the grid-sampling and the UV-blanketing mechanisms.

## 3. Coelho, P., Barbuy, B., Meléndez, J., Schiavon, R. P., & Castilho, B. V. 2005 — "A library of high resolution synthetic stellar spectra from 300 nm to 1.8 μm with solar and α-enhanced composition"

- **Citation:** Coelho, P., Barbuy, B., Meléndez, J., Schiavon, R. P., & Castilho, B. V. 2005, A&A,
  443, 735–746.
- **Stable identifier:** DOI `10.1051/0004-6361:20053511` (arXiv:astro-ph/0505511)
- **First public:** 2005-05-25 (arXiv v1); journal online 2005-11-04, print 2005-11.
- **How verified:** arXiv API for `astro-ph/0505511` returned
  `<published>2005-05-25T13:56:38Z</published>`; the arXiv abs page submission history shows
  "[v1] Wed, 25 May 2005"; Crossref for the DOI returned `published-online = 2005-11-04`,
  `published-print = 2005-11`; OpenAlex returned `publication_date = 2005-11-01`. Earliest
  verifiable public date 2005-05-25, strictly before T0.
- **What it supports:** The reference case for the candidate mechanism that composition changes the
  emergent spectrum in a wavelength-dependent way: a synthetic grid spanning the near-ultraviolet
  to the near-infrared (300 nm – 1.8 μm) computed for both solar and α-enhanced ([α/Fe] = 0.4)
  mixtures, i.e. the same kind of composition axis as the ordered metallicity index in the bundled
  grid.

## 4. Casagrande, L., Ramírez, I., Meléndez, J., Bessell, M., & Asplund, M. 2010 — "An absolutely calibrated T_eff scale from the infrared flux method"

- **Citation:** Casagrande, L., Ramírez, I., Meléndez, J., Bessell, M., & Asplund, M. 2010, A&A,
  512, A54.
- **Stable identifier:** DOI `10.1051/0004-6361/200913204` (arXiv:1001.3142)
- **First public:** 2010-01-18 (arXiv v1); journal print 2010-03, online 2010-04-01.
- **How verified:** arXiv API for `1001.3142` returned `<published>2010-01-18T21:15:09Z</published>`;
  Crossref for the DOI returned `published-print = 2010-03`, `published-online = 2010-04-01`,
  `created = 2010-01-21`; OpenAlex (via Semantic Scholar cross-identifier) gives
  `publicationDate = 2010-01-18`. Earliest verifiable public date 2010-01-18, strictly before T0.
- **What it supports:** The standard statement of why temperature sensitivity differs between the
  ultraviolet and the infrared: for stars hotter than about 4200 K the infrared longward of about
  1.2 μm lies in the Rayleigh–Jeans part of the SED, "a region largely dominated by the continuum
  which linearly depends on T_eff and thus only mildly on model atmospheres", whereas the
  ultraviolet is line-dominated and strongly model-dependent. This supports the candidate mechanism
  behind a temperature-ratio reversal: a T_eff ratio measured in the line-blanketed UV does not
  extrapolate to the nearly blanketing-free Rayleigh–Jeans regime.

## 5. Marr, J. M., & Wilkin, F. P. 2012 — "A better presentation of Planck's radiation law"

- **Citation:** Marr, J. M., & Wilkin, F. P. 2012, American Journal of Physics, 80(5), 399–405.
- **Stable identifier:** DOI `10.1119/1.3696974` (arXiv:1109.3822)
- **First public:** 2011-09-17 (arXiv v1); journal online 2012-04-17, print 2012-05-01.
- **How verified:** arXiv API for `1109.3822` returned `<published>2011-09-17T23:41:22Z</published>`;
  the arXiv abs page shows the same date; Crossref for the DOI returned
  `published-online = 2012-04-17`, `published-print = 2012-05-01`; OpenAlex returned
  `publication_date = 2012-04-17`. Earliest verifiable public date 2011-09-17, strictly before T0.
- **What it supports:** The methods point that "the location of the peak of the Planck curve depends
  on the choice of the independent variable in the plot", and that photon statistics (photon number
  and mean photon energy) are the more meaningful description; this supports the candidate
  alternative explanation that an apparent peak or boundary in a **photon-flux** spectrum is a
  representation-dependent feature, and it argues against using the peak location as a direct
  temperature diagnostic.

## 6. Bessell, M., & Murphy, S. 2012 — "Spectrophotometric Libraries, Revised Photonic Passbands, and Zero Points for UBVRI, Hipparcos, and Tycho Photometry"

- **Citation:** Bessell, M., & Murphy, S. 2012, PASP, 124(912), 140–157.
- **Stable identifier:** DOI `10.1086/664083` (arXiv:1112.2698)
- **First public:** 2011-12-12 (arXiv v1); journal print 2012-02.
- **How verified:** arXiv API for `1112.2698` returned `<published>2011-12-12T00:37:18Z</published>`;
  Crossref for the DOI returned `issued = 2012-02`, `published-print = 2012-02`; OpenAlex returned
  `publication_date = 2012-02-01`. Earliest verifiable public date 2011-12-12, strictly before T0.
- **What it supports:** The operational distinction between photon-counting ("photonic") and
  energy-integrating response functions and the explicit wavelength factor that converts between
  them (the paper's appendix converts between f_lambda, f_nu and photon-flux integrals and states
  that in broadband work the mean photon flux is proportional to the mean energy flux);
  it also states that response functions "must be well sampled and smooth" and warns about the
  confusion caused by mixing photonic and energy passbands. This supports the candidate
  alternative explanation that a sharp feature in a normalized response series can arise from a
  conversion/registration convention (per-unit-wavelength vs per-unit-frequency, photon vs energy
  weighting) rather than from atmosphere physics.

## 7. Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. 2017 — "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"

- **Citation:** Roberts, D. R., et al. 2017, Ecography, 40(8), 913–929.
- **Stable identifier:** DOI `10.1111/ecog.02881`
- **First public:** 2016-12-08 (Crossref `created`, corroborated by OpenAlex); issue online
  2017-03-03, print 2017-08.
- **How verified:** Crossref for the DOI returned `created = 2016-12-08T23:52:39Z`,
  `published-online = 2017-03-03`, `published-print = 2017-08`; OpenAlex returned
  `publication_date = 2016-12-08`; the Crossref abstract text confirms the paper's content (block
  cross-validation for dependent data). Earliest verifiable public date 2016-12-08, strictly before
  T0.
- **What it supports:** The methods justification for validating on **contiguous blocks** rather
  than randomly sampled points when data have dependence structure: uncorrected random
  cross-validation "result[s] in serious underestimation of predictive error" because dependence in
  the data persists in the residuals, and block cross-validation is the remedy. This supports the
  candidate method of holding out contiguous wavelength bands or complete temperature slices, as
  flagged in `DATA_GUIDE.md`, and argues against random-point validation as a sound baseline.

---

## Mechanism map (background only — not evidence about this dataset)

| Candidate mechanism | Supported / ruled out | Sources |
|---|---|---|
| Metallicity / α-enhancement changes emergent flux in a wavelength-dependent way (UV line blanketing, back-warming, redistribution, sign switch) | Supported | 2, 3 |
| Reversal of the flux ratio between two T_eff values between UV (Wien/line-blanketed) and IR (Rayleigh–Jeans, nearly blanketing-free) | Supported | 4 (with 2 for the blanketing leg) |
| Photon-flux vs energy-flux conversion (factor of wavelength) shifts/creates the apparent peak | Supported | 1, 5, 6 |
| A sharp feature in a *normalized* response series is a convention/registration artifact rather than atmosphere physics | Supported as an alternative explanation | 5, 6 |
| Random-point validation is invalid for smooth, dependent spectra; contiguous-band hold-out is required | Supported | 7 |
| A "peak wavelength" of a photon-flux spectrum can be used directly as a temperature diagnostic | Ruled out as a reading of the data | 1, 5 |

## Additional verified pre-T0 sources (not relied upon above)

These were also date-verified during the search and are available if a stronger or more classical
citation is wanted, but they are not needed for the interpretations above.

- **Planck-adjacent classical blanketing reference — Kurucz, R. L. 1979**, "Model atmospheres for G,
  F, A, B, and O stars", ApJS, 40, 1. DOI `10.1086/190589` (ADS bibcode `1979ApJS...40....1K`).
  First public: 1979-05 (Crossref `issued`/`published-print` = 1979-05; OpenAlex
  `publication_date` = 1979-05-01). Supports: line blanketing via opacity distribution functions
  and the resulting changes to fluxes and colours, including a ~2% V-flux error from missing
  molecular lines.
- **Infrared flux method origin — Blackwell, A. E., Shallis, M. J., & Selby, M. J. 1979**, "The
  infrared flux method for determining stellar angular diameters and effective temperatures",
  MNRAS, 188, 847–862. DOI `10.1093/mnras/188.4.847` (ADS bibcode `1979MNRAS.188..847B`). First
  public: 1979-10-01 (Crossref `issued`/`published-print` = 1979-10-01; OpenAlex
  `publication_date` = 1979-10-01; ADS bibcode year 1979 confirmed through the ADS scan service
  `https://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1979MNRAS.188..847B`). Supports:
  the infrared flux method and the historical observation that model-atmosphere predictions
  disagreed with infrared photometry, i.e. that infrared and optical/UV flux calibrations carry
  different systematic behaviour.
- **Coelho, P. R. T. 2014**, "A new library of theoretical stellar spectra with scaled-solar and
  α-enhanced mixtures", MNRAS, 440, 1027–1043. DOI `10.1093/mnras/stu365` (arXiv:1404.3243). First
  public: 2014-04-11 (arXiv v1, `<published>2014-04-11T22:59:48Z</published>`; arXiv abs page
  submission history; Crossref print 2014-04-01). Supports: an explicit caution that
  α-enhancement results depend on the abundance convention — two α-enhanced families per scaled-solar
  iron abundance, for "fixed iron" and "fixed Z" — relevant whenever an abstract "metallicity
  index" is interpreted.
- **Gray, D. F. 2005**, *The Observation and Analysis of Stellar Photospheres*, 3rd ed., Cambridge
  University Press (ISBN 9780521851862). DOI `10.1017/cbo9781316036570`. First public: 2005-11-28
  (OpenLibrary record for ISBN 9780521851862, `publish_date = November 28, 2005`; Crossref
  `issued`/`published-print` = 2005-11-17). Supports: standard textbook treatment of continuous
  opacity, line blanketing, and the relation between observed flux distributions and model
  photospheres.

## Excluded (date not verifiable as pre-T0)

The following candidate sources were encountered and are **not used**, because their first public
date is on or after 2023-01-01 (or cannot be pinned to a stable pre-T0 document):

- **"Thermodynamics of blackbody radiation in nonlinear electrodynamics"**, arXiv:2302.04367.
  First public 2023-02-08 (arXiv API `<published>`), i.e. after T0. Excluded; would otherwise have
  been a candidate for the photon-vs-energy peak discussion.
- **"Mismatching in Planck's, Stephan-Boltzmann and Wine's displacement law for blackbody
  radiation"**, SSRN preprint, DOI `10.2139/ssrn.5495042`. Crossref `issued` = 2025 — after T0.
  Excluded.
- **NIST CODATA "Fundamental Physical Constants" web page**
  (`https://physics.nist.gov/cuu/Constants/index.html`). The page is continuously updated and its
  version history is not exposed as a dated, citable document; the constant values it currently
  serves are the 2022 adjustment. Its first-public date as a stable, citable artifact cannot be
  verified as strictly before 2023-01-01, so it is not used. (The numerical constants needed here
  are instead covered by sources 1 and 5.)