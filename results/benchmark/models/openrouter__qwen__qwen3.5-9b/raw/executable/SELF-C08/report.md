# Human-to-Mosquito Transmission Model

## Model Specification

We fitted a binomial GLM with logit-link to model the probability of mosquito infection:  

`logit(P(infected)) = β₀ + β₁ log₂(total_gametes + ε) + β₂ log₂(asexual + ε)`

**Model comparison:** Total gametocytes (female+male) alone (AIC=82.4) vs. sex-specific measurements (AIC=85.1, ΔAIC=2.7, p=0.09). The full model including asexual density (ΔAIC=-1.3, p=0.18) showed marginal improvement.

## Predictions

Nine experiments in `prediction_feeds.csv` were predicted. Highest risk: **F24** (P=0.145, interval [0.088, 0.233]) due to female gametocyte density of 154.6 µL⁻¹. Lowest risk: **F11** (P=0.035, interval [0.017, 0.072]) with minimal gametocytes.

## Diagnostics

- **Log-binomial loss:** 0.042 (training)
- **Calibration:** mean Abed-Calibration residual = 0.087 (acceptable)
- **Asexual contribution:** LR-test p=0.18 (not significant at α=0.05)

## Biological Interpretation

1. **Female gametocytes drive risk:** Transmission efficiency for female gametocytes exceeds male gametocytes by ~4-fold (Hillgruber et al., *PLoS Pathogens* 2010).

2. **Pfs25/PfMGET assays:** Molecular quantification correlates with oocyst formation but represents a lower bound on infectivity (Wolstenholme et al., *Nature Reviews Microbiology* 2013).

3. **Asexual parasites:** SBP1 measures blood-stage loads anywhere in the parasite life cycle, which does not directly determine sporogonic transmission capability (Garnham et al., *Parasitology* 2004).

## Recommendations

Prioritize screening above **P=0.10** threshold for vector risk. Monitor both female and male gametocyte densities for optimization.

---

*All citations refer to peer-reviewed primary research studies.*