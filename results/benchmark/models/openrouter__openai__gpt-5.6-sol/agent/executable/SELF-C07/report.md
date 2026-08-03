# Delayed female-gametocyte model

I treated the supplied asexual density, *A(t)*, as fixed and piecewise-linearly interpolated. The selected non-negative transit model has five sequestered immature compartments: `I1'=pA-kI1`, `Ij'=k(I[j-1]-Ij)` (j=2,...,5), and circulating mature females `M'=kI5-mu M`, with `k=5/T`. States were zero at day 0. Treatment therefore acts only through the supplied fall in *A(t)*, with no additional gametocytocidal term. Multiple transit compartments approximate a distributed developmental delay. Immature stages I-IV are known to sequester before mature stage V returns to blood; Cao et al. estimated about 8.4 days of sequestration [1].

Detections used an independent lognormal likelihood with fitted log-SD 0.148; ND contributed the probability below 2/mL rather than being set to zero. DOP853 adaptive integration used relative tolerance 1e-08 and absolute tolerance 1e-10. Maximum likelihood estimates were effective production **0.010084/day**, mean maturation **6.17 days**, and mature clearance **0.0708/day**. Thus mature half-life was **9.79 days** (approximate 95% observed-information interval **2.17-44.16 days**). Reuling et al. reported a female Pfs25-based half-life of 5.1 days (95% CI 4.1-6.1), providing biological context rather than external fitting data [2].

Training RMSE was 0.069 log10 units. A one-compartment exponential maturation alternative had AIC 198.6, versus 134.0 for the five-stage model (Delta=64.6), supporting a narrower delay. Requested predictions are in `results.json`; the dense fitted curve is in `trajectory.csv`.

This single short profile weakly separates production, delay shape, and mature loss: continued production from sequestered cohorts can mimic slow clearance. The interval conditions on five stages and the fixed driver, so it excludes structural/driver uncertainty. Assay RNA may also not equal viable-cell abundance.

## References
1. Cao et al. (2019), *eLife* 8:e49058. https://doi.org/10.7554/eLife.49058
2. Reuling et al. (2018), *eLife* 7:e31549. https://doi.org/10.7554/eLife.31549
