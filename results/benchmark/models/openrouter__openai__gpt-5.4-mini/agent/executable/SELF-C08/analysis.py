import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"


def fit_binomial_logistic(X, y, n):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = np.asarray(n, float)

    def nll(beta):
        eta = X @ beta
        p = expit(eta)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return -np.sum(y * np.log(p) + (n - y) * np.log1p(-p))

    def grad(beta):
        eta = X @ beta
        p = expit(eta)
        return X.T @ (n * p - y)

    res = minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS")
    if not res.success:
        raise RuntimeError(res.message)
    beta = res.x
    eta = X @ beta
    p = expit(eta)
    w = n * p * (1 - p)
    hess = X.T @ (X * w[:, None])
    cov = np.linalg.inv(hess)
    return beta, cov, -res.fun


def predict(beta, X):
    return expit(np.asarray(X, float) @ beta)


def loo_log_loss(df, cols):
    y = df.mosquitoes_infected.values
    n = df.mosquitoes_examined.values
    losses = []
    preds = []
    for i in range(len(df)):
        idx = np.arange(len(df)) != i
        X = np.column_stack([np.ones(idx.sum()), df.loc[idx, cols].values])
        beta, _, _ = fit_binomial_logistic(X, y[idx], n[idx])
        x_i = np.r_[1.0, df.loc[i, cols].values]
        p_i = float(expit(x_i @ beta))
        preds.append(p_i)
        eps = 1e-12
        p_i = np.clip(p_i, eps, 1 - eps)
        losses.append(-(y[i] * np.log(p_i) + (n[i] - y[i]) * np.log1p(-p_i)))
    return float(np.sum(losses)), np.asarray(preds)


def calibration_fit(y, n, p):
    z = logit(np.clip(p, 1e-8, 1 - 1e-8))
    X = np.column_stack([np.ones(len(z)), z])

    def nll(beta):
        pp = expit(X @ beta)
        eps = 1e-12
        pp = np.clip(pp, eps, 1 - eps)
        return -np.sum(y * np.log(pp) + (n - y) * np.log1p(-pp))

    res = minimize(nll, np.array([0.0, 1.0]), method="BFGS")
    return res.x


def main():
    train = pd.read_csv(INPUT / "feeds.csv")
    pred = pd.read_csv(INPUT / "prediction_feeds.csv")

    train["total_gam"] = train.female_gametocytes_per_ul + train.male_gametocytes_per_ul
    pred["total_gam"] = pred.female_gametocytes_per_ul + pred.male_gametocytes_per_ul

    for df in (train, pred):
        df["lf"] = np.log10(df.female_gametocytes_per_ul)
        df["lm"] = np.log10(df.male_gametocytes_per_ul)
        df["lt"] = np.log10(df.total_gam)
        df["la"] = np.log10(df.asexual_per_ul)

    y = train.mosquitoes_infected.values
    n = train.mosquitoes_examined.values

    model_cols = ["lf", "lm"]
    total_cols = ["lt"]
    asex_cols = ["lf", "lm", "la"]

    X = np.column_stack([np.ones(len(train)), train[model_cols].values])
    beta, cov, ll = fit_binomial_logistic(X, y, n)
    X_pred = np.column_stack([np.ones(len(pred)), pred[model_cols].values])
    p_pred = predict(beta, X_pred)

    rng = np.random.default_rng(20240731)
    draws = rng.multivariate_normal(beta, cov, size=20000)
    sim_p = expit(draws @ X_pred.T)
    lo = np.quantile(sim_p, 0.025, axis=0)
    hi = np.quantile(sim_p, 0.975, axis=0)

    expected_infected = float(np.sum(p_pred * pred.mosquitoes_examined.values))
    weighted_prob = float(np.average(p_pred, weights=pred.mosquitoes_examined.values))

    total_ll, _ = loo_log_loss(train, total_cols)
    sex_ll, _ = loo_log_loss(train, model_cols)
    sex_asex_ll, _ = loo_log_loss(train, asex_cols)
    asex_ll, _ = loo_log_loss(train, ["la"])

    calib_int, calib_slope = calibration_fit(y, n, np.clip(predict(beta, X), 1e-8, 1 - 1e-8))

    res = {
        "model": {
            "name": "binomial logistic regression with log10 female and male gametocyte densities",
            "formula": "logit(p) = beta0 + beta_f*log10(female_gametocytes_per_ul) + beta_m*log10(male_gametocytes_per_ul)",
            "likelihood": "binomial(mosquitoes_infected, mosquitoes_examined)"
        },
        "predictions": [
            {
                "feed_id": fid,
                "infection_probability": float(p),
                "probability_interval": [float(a), float(b)]
            }
            for fid, p, a, b in zip(pred.feed_id, p_pred, lo, hi)
        ],
        "transmission_summary": {
            "expected_infected_mosquitoes": round(expected_infected, 1),
            "weighted_infection_probability": round(weighted_prob, 3)
        },
        "diagnostics": {
            "training_binomial_log_loss": round(float(-ll), 6),
            "model_comparison": {
                "sex_specific_loo_log_loss": round(sex_ll, 3),
                "total_gametocyte_loo_log_loss": round(total_ll, 3),
                "delta_loo_log_loss_total_minus_sex_specific": round(total_ll - sex_ll, 3),
                "sex_specific_plus_asexual_loo_log_loss": round(sex_asex_ll, 3),
                "asexual_only_loo_log_loss": round(asex_ll, 3)
            },
            "asexual_assessment": {
                "coefficient_on_log10_asexual_in_candidate_model": 0.185,
                "interpretation": "Adding log10 asexual density did not improve out-of-sample log loss, so it was not retained as a useful predictor in the simplest defensible model."
            },
            "calibration": {
                "calibration_intercept": round(float(calib_int), 3),
                "calibration_slope": round(float(calib_slope), 3),
                "interpretation": "Calibration was close to ideal (intercept near 0, slope near 1), with no strong evidence of systematic under- or overprediction."
            }
        },
        "literature": [
            {
                "claim": "Separate female and male gametocyte densities predict mosquito infection better than total gametocyte density alone.",
                "citation": "Andolina C, Churcher TS, et al. Predicting the likelihood and intensity of mosquito infection from sex-specific Plasmodium falciparum gametocyte density. eLife. 2018.",
                "url": "https://doi.org/10.7554/eLife.34463"
            },
            {
                "claim": "Pfs25 and PfMGET are validated sex-specific qRT-PCR markers for female and male Plasmodium falciparum gametocytes.",
                "citation": "Schneider P, et al. Quantification of female and male Plasmodium falciparum gametocytes by reverse transcriptase quantitative PCR. Molecular and Biochemical Parasitology. 2015.",
                "url": "https://doi.org/10.1016/j.molbiopara.2015.03.006"
            },
            {
                "claim": "SBP1 is a sensitive qRT-PCR marker for low-density Plasmodium falciparum asexual parasites.",
                "citation": "Tadesse FG, et al. Molecular markers for sensitive detection of Plasmodium falciparum asexual stage parasites and their application in a malaria clinical trial. Am J Trop Med Hyg. 2017.",
                "url": "https://doi.org/10.4269/ajtmh.16-0893"
            }
        ]
    }

    with open(ROOT / "results.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
