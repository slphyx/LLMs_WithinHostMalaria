# Modeling Human-to-Mosquito Transmission Dynamics

## Model Architecture & Evaluation
We evaluated binomial logistic GLMs parameterizing the infection probability $p_i$ of mosquitoes in direct membrane feeding experiments (DMFAs). Mosquito infections follow a Binomial likelihood $y_i \sim \text{Binomial}(n_i, p_i)$, where $n_i$ mosquitoes are examined and $y_i$ develop oocysts.

Comparing a total-gametocyte model ($\text{logit}(p) = \beta_0 + \beta_T \log(G_{\text{female}} + G_{\text{male}})$) against a sex-specific model ($\text{logit}(p) = \beta_0 + \beta_F \log G_{\text{female}} + \beta_M \log G_{\text{male}}$) demonstrates that separate sex densities provide superior fit ($\Delta\text{AIC} = 11.76$; Likelihood Ratio Test $\chi^2_1 = 13.76, p = 2.08 \times 10^{-4}$). Both female ($\hat{\beta}_F = 0.878 \pm 0.114, p < 0.001$) and male ($\hat{\beta}_M = 0.547 \pm 0.102, p < 0.001$) gametocyte densities independently contribute to transmission probability. Because fertilization in the mosquito midgut requires at least one microgamete and one macrogamete, male availability acts as a distinct limiting bottleneck when gametocyte sex ratios fluctuate.

## Role of Asexual Parasite Density
Adding log circulating asexual parasite density (measured via SBP1 qRT-PCR) to the sex-specific model does not yield predictive improvement ($\Delta\text{AIC} = +1.44$; LRT $\chi^2_1 = 0.56, p = 0.453$). While asexual parasitemia correlates with gametocyte production upstream, asexual stages do not enter the mosquito or participate in fertilization. Therefore, asexual parasite density serves as a non-causal correlated biomarker rather than a direct determinant of mosquito infection.

## Calibration and Prediction Uncertainty
The sex-specific binomial model demonstrates excellent calibration across the 27 experimental feeds, with no evidence of overdispersion or lack of fit (residual deviance = 15.55 on 24 d.f., $p = 0.904$; Pearson $\chi^2 = 14.81, p = 0.926$). The training binomial log loss is 0.15018 per mosquito.

Prediction uncertainty for 9 test feeds was quantified using the delta method on the logit scale ($95\%\text{ CI} = \text{inv-logit}(\hat{\eta} \pm 1.96 \cdot \text{SE}(\hat{\eta}))$), validated via 2,000 parametric bootstrap iterations. Across 823 examined mosquitoes in the prediction set, the model estimates 89.53 expected infected mosquitoes, corresponding to a weighted infection probability of 0.1088 (10.88%).

## Biological Literature Justification
1. **Sex-Specific Molecular Assays**: [Meerstein-Kessel et al. (2018)](https://doi.org/10.1186/s12936-018-2584-y) validated multiplex qRT-PCR targeting Pfs25 for female gametocytes and PfMGET for male gametocytes, establishing reliable quantification of gametocyte sex ratio.
2. **Assay Streams & Transmission Dynamics**: [Bradley et al. (2021a)](https://doi.org/10.1093/cid/ciaa1853) demonstrated that Pfs25 and PfMGET gametocyte densities directly predict mosquito infectivity in feeding assays, while SBP1 specifically quantifies non-infectious asexual parasites.
3. **Sexual Stage Bottlenecks**: [Bradley et al. (2021b)](https://doi.org/10.1038/s41467-021-22573-7) showed that functional transmission depends on male and female gametocyte availability, confirming that total gametocyte counts mask essential sex-ratio dynamics governing midgut fertilization success.
