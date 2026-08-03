# Human-to-mosquito transmission from sex-specific gametocyte densities

## Model

Each feed contributes `mosquitoes_infected ~ Binomial(mosquitoes_examined, p)`, so the
binomial numerator and denominator (not observed proportions) enter the likelihood. The
mean model is a logistic GLM linear in log10 density, giving the saturating dose–response
seen in membrane-feeding studies ([Bradley et al., *eLife* 2018](https://doi.org/10.7554/eLife.34463)):

logit(p) = −5.497 + 2.022·log10(female/µL) + 1.259·log10(male/µL)
(SEs 0.389, 0.262, 0.235). Fitted by maximum likelihood with `scipy.optimize` (multi-start BFGS).

## Sex-specific versus total gametocytes

The sex-specific model beats the total-gametocyte model `logit(p)=b0+b1·log10(female+male)`
on every criterion: AIC 778.5 vs 790.3, leave-one-feed-out binomial log loss 0.1514 vs
0.1563, and a likelihood-ratio test of the extra male-specific term χ²=13.8, df=1, p<0.001.
A female-only model is worst (LOO 0.1606). The female slope is ~1.6× the male slope,
consistent with infection being driven mainly by female gametocytes while male supply still
limits transmission at low densities ([Bradley 2018](https://doi.org/10.7554/eLife.34463)).
Because Pfs25 and PfMGET qRT-PCR quantify the two sexes with independent transcript targets
([Santolamazza et al., *J Infect Dis* 2017](https://doi.org/10.1093/infdis/jix237)), summing
them discards real information; log10 female and log10 male are correlated (r=0.81) but both
coefficients remain estimable, with correlated uncertainty.

## Asexual density

Adding log10 asexual density gives χ²=0.56, df=1, p=0.45 and *worsens* out-of-sample log
loss (0.1522 vs 0.1514), so it was excluded. Asexual density is correlated with total
gametocytaemia (r=0.73, log10 scale); asexual stages are not infectious to mosquitoes, so any
residual association would be a proxy for concurrent gametocyte carriage, not a causal
pathway ([Ouédraogo et al., *J Infect Dis* 2016](https://doi.org/10.1093/infdis/jiv370)).
Molecular density is a biomarker of transcript abundance, not a count of viable gametocytes
([Stone et al., *Malar J* 2017](https://doi.org/10.1186/s12936-016-1584-z)).

## Uncertainty, calibration and fit

Training binomial log loss per mosquito 0.1502. Prediction intervals are 95% Monte-Carlo
intervals for the mean probability from 40,000 draws of the asymptotic sampling distribution
of the coefficients. Pearson χ²=14.4 on 24 df (dispersion 0.60, p=0.94) and a beta-binomial
extension is not supported (LRT p=0.36, φ≈670), so binomial sampling variance suffices — the
intervals do not need overdispersion inflation. Grouped calibration in three equal-count
bins of fitted probability: 0.0003 vs 0.000 observed, 0.0071 vs 0.008, 0.174 vs 0.183
(Hosmer–Lemeshow-type statistic 0.35, p=0.55). No lack of fit is detected.

## Predictions

Nine held-out feeds range from p≈0.0001 (F11, F15) to 0.55 (F24, 95% CI 0.46–0.64).
Across 823 mosquitoes the model expects 89.5 infections, a denominator-weighted infection
probability of 0.109. Caveats: only 27 feeds, most with zero infections, so intervals in the
sparse low-density region rely on the parametric logit-linear extrapolation; these are
synthetic data, and the coefficients are associations within this design, not identified
mechanisms.

*(~470 words)*
