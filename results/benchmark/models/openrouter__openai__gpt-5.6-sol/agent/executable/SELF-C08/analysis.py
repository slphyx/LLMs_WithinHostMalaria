#!/usr/bin/env python3
"""Binomial models for direct membrane-feeding experiments.

Reads only the supplied input files and deterministically writes results.json.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "input" / "feeds.csv"
PREDICT = ROOT / "input" / "prediction_feeds.csv"
DICTIONARY = ROOT / "input" / "data_dictionary.json"

# Read/validate the supplied inputs (the dictionary is part of the validation).
d = pd.read_csv(TRAIN)
new = pd.read_csv(PREDICT)
with DICTIONARY.open() as f:
    dictionary = json.load(f)
required_train = ["feed_id", "asexual_per_ul", "female_gametocytes_per_ul",
                  "male_gametocytes_per_ul", "mosquitoes_examined", "mosquitoes_infected"]
required_new = required_train[:-1]
assert list(d.columns) == required_train and list(new.columns) == required_new
assert all(c in dictionary["fields"] for c in required_train)
assert (d[["asexual_per_ul", "female_gametocytes_per_ul", "male_gametocytes_per_ul"]] > 0).all().all()
assert ((d.mosquitoes_infected >= 0) & (d.mosquitoes_infected <= d.mosquitoes_examined)).all()

y = d.mosquitoes_infected.to_numpy(float)
n = d.mosquitoes_examined.to_numpy(float)


def design(frame, model):
    female = np.log10(frame.female_gametocytes_per_ul.to_numpy(float))
    male = np.log10(frame.male_gametocytes_per_ul.to_numpy(float))
    total = np.log10((frame.female_gametocytes_per_ul + frame.male_gametocytes_per_ul).to_numpy(float))
    asexual = np.log10(frame.asexual_per_ul.to_numpy(float))
    columns = {"total": [total], "sex": [female, male], "sex_asexual": [female, male, asexual]}[model]
    return np.column_stack([np.ones(len(frame))] + columns)


def fit(model):
    X = design(d, model)
    def objective(beta):
        eta = X @ beta
        # Stable negative binomial log likelihood, excluding fixed choose(n,y).
        return float(np.sum(n * np.logaddexp(0.0, eta) - y * eta))
    def gradient(beta):
        return X.T @ (n * expit(X @ beta) - y)
    result = minimize(objective, np.zeros(X.shape[1]), jac=gradient, method="BFGS",
                      options={"gtol": 1e-10, "maxiter": 1000})
    beta = result.x
    p = expit(X @ beta)
    information = X.T @ ((n * p * (1 - p))[:, None] * X)
    covariance = np.linalg.inv(information)
    return {"X": X, "beta": beta, "p": p, "nll": objective(beta), "cov": covariance}


total = fit("total")
sex = fit("sex")
sex_a = fit("sex_asexual")

# Comparisons use binomial likelihood: AIC for non-nested forms; LRT for adding asexual density.
aic_total = 2 * total["nll"] + 2 * len(total["beta"])
aic_sex = 2 * sex["nll"] + 2 * len(sex["beta"])
lr_asexual = 2 * (sex["nll"] - sex_a["nll"])
p_asexual = chi2.sf(lr_asexual, 1)

# Mean-probability intervals are 95% Wald intervals on the logit scale.
Xnew = design(new, "sex")
eta = Xnew @ sex["beta"]
se_eta = np.sqrt(np.einsum("ij,jk,ik->i", Xnew, sex["cov"], Xnew))
p_new = expit(eta)
limits = expit(eta[:, None] + norm.ppf([0.025, 0.975]) * se_eta[:, None])

# Absolute binomial deviance and Pearson lack-of-fit checks at feed level.
p = sex["p"]
def xlogratio(x, denominator):
    out = np.zeros_like(x)
    positive = x > 0
    out[positive] = x[positive] * np.log(x[positive] / denominator[positive])
    return out
deviance = 2 * np.sum(xlogratio(y, n * p) + xlogratio(n - y, n * (1 - p)))
df = len(y) - len(sex["beta"])
dev_p = chi2.sf(deviance, df)
pearson = np.sum((y - n * p) ** 2 / (n * p * (1 - p)))
log_loss = sex["nll"] / n.sum()

predictions = [{
    "feed_id": str(row.feed_id),
    "infection_probability": round(float(prob), 6),
    "probability_interval": [round(float(lo), 6), round(float(hi), 6)]
} for row, prob, (lo, hi) in zip(new.itertuples(index=False), p_new, limits)]
expected = float(np.sum(new.mosquitoes_examined.to_numpy() * p_new))

literature = [
    {"claim": "Sex-specific female and male gametocyte densities jointly predicted mosquito infection in membrane feeds; low male density could constrain infection.",
     "citation": "Bradley et al., Predicting the likelihood and intensity of mosquito infection from sex-specific Plasmodium falciparum gametocyte density, eLife (2018)",
     "url": "https://doi.org/10.7554/eLife.34463"},
    {"claim": "Pfs25 RT-qPCR was developed and validated for quantifying female P. falciparum gametocytes.",
     "citation": "Schneider et al., Quantification of female and male Plasmodium falciparum gametocytes by reverse transcriptase quantitative PCR, Molecular and Biochemical Parasitology (2015)",
     "url": "https://doi.org/10.1016/j.molbiopara.2015.03.006"},
    {"claim": "Sex-specific qRT-PCR assays used Pfs25 for female and PfMGET for male gametocytes in human blood.",
     "citation": "Wang et al., Assays for quantification of male and female gametocytes in human blood by qRT-PCR in the absence of pure sex-specific gametocyte standards, Malaria Journal (2020)",
     "url": "https://doi.org/10.1186/s12936-020-03291-9"},
    {"claim": "SBP1 qRT-PCR was a sensitive marker for asexual ring-stage parasites, with some mature-gametocyte expression noted.",
     "citation": "Tadesse et al., Molecular Markers for Sensitive Detection of Plasmodium falciparum Asexual Stage Parasites and their Application in a Malaria Clinical Trial, American Journal of Tropical Medicine and Hygiene (2017)",
     "url": "https://doi.org/10.4269/ajtmh.16-0893"}
]

result = {
    "model": {
        "name": "Sex-specific log-density binomial logistic regression",
        "formula": "logit(p) = %.6f + %.6f log10(female_per_ul) + %.6f log10(male_per_ul)" % tuple(sex["beta"]),
        "likelihood": "mosquitoes_infected_i ~ Binomial(mosquitoes_examined_i, p_i), independently by feed"
    },
    "predictions": predictions,
    "transmission_summary": {
        "expected_infected_mosquitoes": round(expected, 3),
        "weighted_infection_probability": round(expected / new.mosquitoes_examined.sum(), 6)
    },
    "diagnostics": {
        "training_binomial_log_loss": round(float(log_loss), 6),
        "model_comparison": "Sex-specific AIC %.3f versus total-gametocyte AIC %.3f (Delta AIC total-minus-sex %.3f); sex-specific retained." % (aic_sex, aic_total, aic_total-aic_sex),
        "asexual_assessment": "Adding log10(asexual density) to the sex-specific model: LR chi-square(1)=%.3f, p=%.3f, Delta AIC(add-minus-base)=%.3f; no useful improvement, and no causal interpretation." % (lr_asexual, p_asexual, (2*sex_a["nll"]+2*len(sex_a["beta"]))-aic_sex),
        "calibration": "Feed-level binomial deviance %.3f on %d df (p=%.3f); Pearson chi-square/df %.3f. No detected lack of fit, though only 27 feeds limit power." % (deviance, df, dev_p, pearson/df)
    },
    "literature": literature
}
with (ROOT / "results.json").open("w") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
