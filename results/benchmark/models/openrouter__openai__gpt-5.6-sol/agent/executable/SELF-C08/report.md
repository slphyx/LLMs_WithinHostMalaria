# Human-to-mosquito transmission

## Model and selection

I fitted feed-level binomial logistic regressions to infected **counts**, with each feed's examined count as its denominator. Densities were log10-transformed because they span orders of magnitude. The selected model was

`logit(p) = -5.496 + 2.022 log10(female/µL) + 1.259 log10(male/µL)`.

This is the simplest candidate retaining sex-specific biology. Its AIC was 778.50, compared with 790.26 for `log10(female + male)` (ΔAIC 11.76). This supports retaining both measurements, not interpreting either coefficient mechanistically. Such a distinction is biologically plausible: sex-specific densities jointly predicted membrane-feed infection, and low male density could constrain infection in [Bradley et al. (2018)](https://doi.org/10.7554/eLife.34463).

Adding log10 asexual density improved log likelihood by only 0.282 (likelihood-ratio χ²(1)=0.564, p=0.453) and worsened AIC by 1.44, so it was excluded. SBP1 is a sensitive ring-stage marker but has some mature-gametocyte expression ([Tadesse et al., 2017](https://doi.org/10.4269/ajtmh.16-0893)); regardless, this observational predictive comparison cannot establish an asexual-stage causal effect.

## Predictions and checking

`results.json` gives predicted mean infection probabilities and 95% coefficient-uncertainty intervals, computed on the logit scale, plus the expected infected count using each requested denominator. Across the nine feeds, 89.529 of 823 mosquitoes are expected infected (weighted probability 0.1088). F24 is a modest extrapolation in female density (154.6/µL versus training maximum 90.7/µL), so its interval may understate structural uncertainty. Intervals describe uncertainty in mean probability conditional on this model; an individual binary outcome remains 0 or 1.

Training per-mosquito binomial log loss was 0.1502. Feed-level deviance was 15.55 on 24 df (p=0.904), and Pearson χ²/df was 0.600: there is no detected overdispersion or lack of fit, although 27 feeds give limited power and the low ratio suggests conservative relative-to-binomial variation rather than extra variation.

## Measurement interpretation

These RNA biomarkers are calibrated molecular densities, not direct counts of viable, mature, or infectious cells. Pfs25 female quantification was validated by [Schneider et al. (2015)](https://doi.org/10.1016/j.molbiopara.2015.03.006); paired Pfs25/PfMGET female/male qRT-PCR assays were developed by [Wang et al. (2020)](https://doi.org/10.1186/s12936-020-03291-9). Thus predictions are associations for these assay streams and feeding conditions, not causal claims or universal transmission probabilities.
