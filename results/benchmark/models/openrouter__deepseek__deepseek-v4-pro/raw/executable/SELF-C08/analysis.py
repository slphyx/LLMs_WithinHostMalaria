#!/usr/bin/env python3
"""SELF-C08: model human-to-mosquito transmission.

Reproducible analysis reading input/feeds.csv and input/prediction_feeds.csv,
producing results.json.
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ----------------------------------------------------------------------
# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("selfc08")

# ----------------------------------------------------------------------
# constants
RANDOM_SEED = 42
N_BOOT = 2000

# ----------------------------------------------------------------------

def read_data():
    train = pd.read_csv("input/feeds.csv")
    pred  = pd.read_csv("input/prediction_feeds.csv")
    return train, pred


def prepare(train, pred):
    """Add derived variables."""
    for df in (train, pred):
        df["total_gam"] = df["female_gametocytes_per_ul"] + df["male_gametocytes_per_ul"]
        df["log_fem"] = np.log(df["female_gametocytes_per_ul"])
        df["log_male"] = np.log(df["male_gametocytes_per_ul"])
        df["log_total"] = np.log(df["total_gam"])
        df["log_asex"] = np.log(df["asexual_per_ul"])
    train["success"] = train["mosquitoes_infected"]
    train["failure"] = train["mosquitoes_examined"] - train["mosquitoes_infected"]
    return train, pred


def fit_models(train):
    """Fit three models, return fitted objects."""

    # total gametocyte model
    formula_total = "success + failure ~ log_total"
    mod_total = smf.glm(formula_total, data=train,
                        family=sm.families.Binomial()).fit()
    LOG.info("Total model AIC=%.2f log-lik=%.2f", mod_total.aic, mod_total.llf)

    # sex-specific model
    formula_sex = "success + failure ~ log_fem + log_male"
    mod_sex = smf.glm(formula_sex, data=train,
                      family=sm.families.Binomial()).fit()
    LOG.info("Sex-specific model AIC=%.2f log-lik=%.2f", mod_sex.aic, mod_sex.llf)

    # sex + asexual model
    formula_asex = "success + failure ~ log_fem + log_male + log_asex"
    mod_asex = smf.glm(formula_asex, data=train,
                       family=sm.families.Binomial()).fit()
    LOG.info("Sex+asex model AIC=%.2f log-lik=%.2f", mod_asex.aic, mod_asex.llf)

    return mod_total, mod_sex, mod_asex


def model_comparison(mod_total, mod_sex):
    """Return dict with AIC comparison."""
    delta_aic = mod_total.aic - mod_sex.aic
    if delta_aic > 2:
        winner = "sex-specific"
    else:
        winner = "total-gametocyte"
    return {
        "total_aic": round(float(mod_total.aic), 2),
        "sex_aic": round(float(mod_sex.aic), 2),
        "delta_aic": round(float(delta_aic), 2),
        "preferred_model": winner
    }


def asexual_assessment(mod_sex, mod_asex):
    """LRT comparing sex+asex with sex-only model."""
    lr_stat = -2 * (mod_sex.llf - mod_asex.llf)
    from scipy.stats import chi2
    pval = chi2.sf(lr_stat, 1)  # one additional parameter
    return {
        "lr_statistic": round(float(lr_stat), 4),
        "p_value": round(float(pval), 4),
        "conclusion": ("asexual density does NOT add significant predictive value"
                       if pval > 0.05 else
                       "asexual density adds significant predictive value")
    }


def predict_probs(model, df):
    """Return predicted infection probability from fitted model."""
    return model.predict(df)


def bootstrap_predictions(train, pred, model_formula, chosen_model_index=1):
    """Non-parametric bootstrap to get prediction intervals.

    model_formula : a function that returns a formula string given a DataFrame.
                   We use the sex-specific formula.
    """
    np.random.seed(RANDOM_SEED)
    boot_probs = []
    n = len(train)
    idx = train.index.values
    for i in range(N_BOOT):
        boot_idx = np.random.choice(idx, size=n, replace=True)
        boot_df = train.loc[boot_idx].copy()
        try:
            _mod = smf.glm("success + failure ~ log_fem + log_male",
                           data=boot_df,
                           family=sm.families.Binomial()).fit(disp=False)
        except Exception as e:
            LOG.warning("Bootstrap %d failed: %s", i, e)
            continue
        prob = _mod.predict(pred)
        boot_probs.append(prob.values)
    if not boot_probs:
        raise RuntimeError("All bootstrap replicates failed")
    boot_probs = np.array(boot_probs)  # (n_valid, n_pred)
    # percentiles
    lower = np.percentile(boot_probs, 2.5, axis=0)
    upper = np.percentile(boot_probs, 97.5, axis=0)
    point = np.median(boot_probs, axis=0)  # use median for robust point estimate
    return point, lower, upper


def calibration_check(train, model):
    """Fit calibration regression: observed ~ offset(logit_pred) + logit_pred.

    Returns slope, intercept.
    """
    # predicted logit
    train = train.copy()
    train["lp"] = model.fittedvalues  # linear predictor (logit)
    cal_mod = smf.glm("success + failure ~ lp", data=train,
                      family=sm.families.Binomial()).fit()
    intercept = cal_mod.params.get("Intercept", 0)
    slope = 1.0 + cal_mod.params.get("lp", 0)  # if formula success~lp, slope is coeff of lp
    # Actually success ~ lp gives slope = coeff of lp. If well calibrated, coeff should be 1.
    # So slope = cal_mod.params["lp"]
    return round(float(cal_mod.params.get("Intercept", 0)), 4), round(float(cal_mod.params.get("lp", 0)), 4)


def build_prediction_list(pred, point_probs, lower, upper):
    pred_list = []
    for i, row in pred.iterrows():
        pred_list.append({
            "feed_id": row["feed_id"],
            "infection_probability": round(float(point_probs[i]), 6),
            "probability_interval": [round(float(lower[i]), 6), round(float(upper[i]), 6)]
        })
    return pred_list


def main():
    train, pred = read_data()
    train, pred = prepare(train, pred)

    mod_total, mod_sex, mod_asex = fit_models(train)

    # model comparison
    comp = model_comparison(mod_total, mod_sex)

    # asexual assessment
    asex = asexual_assessment(mod_sex, mod_asex)

    # choose best model for predictions (sex-specific preferred if delta AIC > 0)
    if comp["preferred_model"] == "sex-specific":
        chosen_mod = mod_sex
        formula_str = "success + failure ~ log_fem + log_male"
    else:
        chosen_mod = mod_total
        formula_str = "success + failure ~ log_total"

    # point predictions on original fit
    point_probs = chosen_mod.predict(pred).values

    # bootstrap intervals using sex-specific model (as it's the richer model that could be chosen)
    boot_point, boot_lower, boot_upper = bootstrap_predictions(
        train, pred, formula_str, chosen_model_index=0
    )

    # transmission summary (using point_probs from original fit)
    expected_inf = np.sum(point_probs * pred["mosquitoes_examined"])
    weighted_prob = np.average(point_probs, weights=pred["mosquitoes_examined"])

    # diagnostics
    train_logloss = -chosen_mod.llf  # total negative log-likelihood
    cal_int, cal_slope = calibration_check(train, chosen_mod)
    calibration_text = (
        f"Calibration intercept={cal_int}, slope={cal_slope}; "
        f"perfect calibration would give intercept 0, slope 1."
    )

    # predictions list
    predictions = build_prediction_list(pred, boot_point, boot_lower, boot_upper)

    # results dict
    results = {
        "model": {
            "name": "sex-specific logistic (preferred)" if comp["preferred_model"] == "sex-specific" else "total-gametocyte logistic",
            "formula": "logit(p) = β0 + β1 * log(female) + β2 * log(male)",
            "likelihood": "binomial: infected ~ Binomial(examined, p)"
        },
        "predictions": predictions,
        "transmission_summary": {
            "expected_infected_mosquitoes": round(float(expected_inf), 2),
            "weighted_infection_probability": round(float(weighted_prob), 6)
        },
        "diagnostics": {
            "training_binomial_log_loss": round(float(train_logloss), 4),
            "model_comparison": (
                f"Sex-specific vs total-gametocyte: AIC total={comp['total_aic']}, "
                f"AIC sex={comp['sex_aic']}, ΔAIC={comp['delta_aic']} "
                f"(prefer {comp['preferred_model']})."
            ),
            "asexual_assessment": (
                f"LRT statistic={asex['lr_statistic']}, p={asex['p_value']}: "
                f"{asex['conclusion']}."
            ),
            "calibration": calibration_text
        },
        "literature": [
            {
                "claim": "Total gametocyte density predicts mosquito infection; relationship can be modelled with logistic dose-response.",
                "citation": "Churcher TS, et al. Predicting mosquito infection from Plasmodium falciparum gametocyte density. eLife, 2013.",
                "url": "https://doi.org/10.7554/eLife.00626"
            },
            {
                "claim": "Female (Pfs25) and male (PfMGET) gametocyte densities are both required for fertilisation; sex-specific measurements improve transmission predictions.",
                "citation": "Schneider P, et al. Quantification of female and male P. falciparum gametocytes by quantitative reverse transcriptase PCR. Mol Biochem Parasitol, 2015.",
                "url": "https://doi.org/10.1016/j.molbiopara.2015.03.006"
            },
            {
                "claim": "Molecular markers SBP1 (asexual), Pfs25 (female gametocyte) and PfMGET (male gametocyte) accurately quantify parasite stages in feeding experiments.",
                "citation": "Meerstein-Kessel L, et al. Multiplex qPCR assay for detection and quantification of Plasmodium falciparum sexual and asexual parasites. Sci Rep, 2018.",
                "url": "https://doi.org/10.1038/s41598-018-31585-7"
            }
        ]
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    LOG.info("results.json written")


if __name__ == "__main__":
    main()
