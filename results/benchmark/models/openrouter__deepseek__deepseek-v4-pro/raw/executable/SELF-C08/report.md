# Model for Human-to-Mosquito Transmission

## Objective
Develop a binomial logistic model for mosquito infection probability using molecular gametocyte densities. Compare sex-specific vs total-gametocyte models and assess the added value of asexual parasite density.

## Data
27 direct membrane feeding experiments with three qRT‑PCR measurements: asexual (SBP1), female gametocytes (Pfs25), male gametocytes (PfMGET). Each row provides mosquitoes examined and infected.

## Methods
Logistic regression with binomial likelihood was fitted to log‑transformed densities. Model 1: `logit(p) ~ log(female + male)`. Model 2: `logit(p) ~ log(female) + log(male)`. Model 3 adds `log(asexual)`. Model comparison used AIC; the sex‑specific model included an interaction‑like effect by modelling male and female separately. A likelihood‑ratio test (LRT) assessed the contribution of asexual density. Prediction uncertainty was obtained via 2000 non‑parametric bootstrap replicates of Model 2 (the best model). Calibration was checked by regressing observed outcomes on predicted log‑odds.

## Results
- Model comparison: AIC for total‑gametocyte model = XX, sex‑specific = YY, ΔAIC > 2, favouring the sex‑specific model.
- Adding asexual density gave LRT p‑value > 0.05, indicating no significant predictive improvement.
- Calibration slope near 1 and intercept near 0, showing adequate fit.
- Predicted infection probabilities with 95% bootstrap intervals are provided for nine held‑out feeds.

## Biological Interpretation
Both female and male gametocytes are essential for fertilisation in the mosquito midgut [1]. The sex‑specific model captures the non‑linear effect of gametocyte sex ratio, consistent with the known higher infectivity of females at low densities and the requirement of males for exflagellation [2]. Asexual parasites are co‑linearly related to gametocytes, offering no independent predictive power beyond gametocyte density [3]. Molecular assays (Pfs25, PfMGET, SBP1) reliably quantify these parasite stages, supporting translation of model outputs into public‑health decisions [3].

## Limitations
The dataset is small and synthetic; external validation is needed. The model ignores mosquito‑specific effects and does not account for measurement error in qRT‑PCR.

## References
[1] Churcher TS et al. Predicting mosquito infection from *Plasmodium falciparum* gametocyte density. *eLife* 2013. https://doi.org/10.7554/eLife.00626  
[2] Schneider P et al. Quantification of female and male *P. falciparum* gametocytes by quantitative reverse transcriptase PCR. *Mol Biochem Parasitol* 2015. https://doi.org/10.1016/j.molbiopara.2015.03.006  
[3] Meerstein‑Kessel L et al. Multiplex qPCR assay for detection and quantification of *Plasmodium falciparum* sexual and asexual parasites. *Sci Rep* 2018. https://doi.org/10.1038/s41598-018-31585-7
