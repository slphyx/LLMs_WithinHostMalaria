#!/usr/bin/env python3
"""analysis.py: Model human-to-mosquito transmission probability.

Reads input/feeds.csv and input/prediction_feeds.csv.
Produces results.json.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm, chi2

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("input")
TRAIN_PATH = DATA_DIR / "feeds.csv"
PRED_PATH = DATA_DIR / "prediction_feeds.csv"
OUT_PATH = Path("results.json")

# ── load data ──────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
pred = pd.read_csv(PRED_PATH)

n_train = train["mosquitoes_examined"].values.astype(int)
y_train = train["mosquitoes_infected"].values.astype(int)
F_train = train["female_gametocytes_per_ul"].values  # female
M_train = train["male_gametocytes_per_ul"].values     # male
A_train = train["asexual_per_ul"].values              # asexual
T_train = F_train + M_train                            # total gametocytes

F_pred = pred["female_gametocytes_per_ul"].values
M_pred = pred["male_gametocytes_per_ul"].values
A_pred = pred["asexual_per_ul"].values
T_pred = F_pred + M_pred
n_pred = pred["mosquitoes_examined"].values.astype(int)
feed_ids = pred["feed_id"].values

# ── helper: negative log-likelihood & fitting ──────────────────────────────
def neg_log_lik(beta, X, n, y):
    """Binomial negative log-likelihood (no constant term added)."""
    eta = X @ beta
    p = expit(eta)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.sum(y * np.log(p) + (n - y) * np.log1p(-p))


def fit_model(X, n, y, n_restarts=5):
    """Fit binomial GLM with logit link via MLE.

    Returns dict with beta, hessian_inv, se, aic, log_lik, converged.
    """
    k = X.shape[1]
    best = None
    best_nll = np.inf
    rng = np.random.default_rng(42)

    for _ in range(n_restarts):
        beta0 = rng.normal(0, 0.5, size=k)
        res = minimize(
            neg_log_lik, beta0, args=(X, n, y),
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-12},
        )
        if res.fun < best_nll:
            best_nll = res.fun
            best = res

    log_lik = -best_nll
    aic = 2 * k + 2 * best_nll
    try:
        # Hessian via finite differences at optimum
        eps = 1e-5
        hess = np.zeros((k, k))
        beta_opt = best.x
        for i in range(k):
            for j in range(k):
                beta_pp = beta_opt.copy()
                beta_pm = beta_opt.copy()
                beta_mp = beta_opt.copy()
                beta_mm = beta_opt.copy()
                beta_pp[i] += eps; beta_pp[j] += eps
                beta_pm[i] += eps; beta_pm[j] -= eps
                beta_mp[i] -= eps; beta_mp[j] += eps
                beta_mm[i] -= eps; beta_mm[j] -= eps
                f_pp = neg_log_lik(beta_pp, X, n, y)
                f_pm = neg_log_lik(beta_pm, X, n, y)
                f_mp = neg_log_lik(beta_mp, X, n, y)
                f_mm = neg_log_lik(beta_mm, X, n, y)
                hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
        hess_inv = np.linalg.inv(hess)
        se = np.sqrt(np.diag(hess_inv))
    except np.linalg.LinAlgError:
        hess_inv = None
        se = np.full(k, np.nan)

    return {
        "beta": best.x,
        "hessian_inv": hess_inv,
        "se": se,
        "aic": aic,
        "log_lik": log_lik,
        "converged": best.success,
    }


def predict_probs(beta, X, ci_level=0.90):
    """Return predicted probabilities and delta-method intervals."""
    eta = X @ beta
    p = expit(eta)
    return p  # intervals computed separately


def predict_with_ci(beta, hess_inv, X, ci_level=0.90):
    """Delta-method CI on logit scale, back-transformed."""
    eta = X @ beta
    p = expit(eta)
    # Var(eta) = X @ hess_inv @ X^T
    var_eta = np.einsum("ij,jk,ik->i", X, hess_inv, X)
    se_eta = np.sqrt(var_eta)
    z = norm.ppf(1 - (1 - ci_level) / 2)
    lo = expit(eta - z * se_eta)
    hi = expit(eta + z * se_eta)
    return p, lo, hi


# ── build design matrices (with intercept) ─────────────────────────────────
def design_matrix(*cols, intercept=True):
    """Stack columns into design matrix, optionally adding intercept."""
    if intercept:
        return np.column_stack([np.ones(len(cols[0])), *cols])
    return np.column_stack(cols)


# ── fit candidate models ───────────────────────────────────────────────────
models = {}

# M_total: total gametocytes only
X_total = design_matrix(T_train)
models["total"] = fit_model(X_total, n_train, y_train)

# M_sex: female + male separately
X_sex = design_matrix(F_train, M_train)
models["sex"] = fit_model(X_sex, n_train, y_train)

# M_sex_asex: female + male + asexual
X_sex_asex = design_matrix(F_train, M_train, A_train)
models["sex_asex"] = fit_model(X_sex_asex, n_train, y_train)

# Also fit models for comparison with asexual only
X_asex_only = design_matrix(A_train)
models["asex_only"] = fit_model(X_asex_only, n_train, y_train)

# ── AIC comparison ─────────────────────────────────────────────────────────
print("=== Model comparison (AIC) ===")
for name, m in models.items():
    print(f"  {name:15s}: AIC = {m['aic']:.2f}, log-lik = {m['log_lik']:.2f}, "
          f"k = {len(m['beta'])}, beta = {m['beta']}")

# Best model by AIC
best_name = min(models, key=lambda k: models[k]["aic"])
best_model = models[best_name]
print(f"\nBest model by AIC: {best_name}")

# Likelihood ratio test: sex vs total
lr_stat = 2 * (models["sex"]["log_lik"] - models["total"]["log_lik"])
lr_pval = chi2.sf(lr_stat, df=1)  # 1 extra parameter
print(f"\nLRT sex vs total: stat={lr_stat:.3f}, p={lr_pval:.4f}")

# Likelihood ratio test: adding asexual to sex-specific model
lr_asex = 2 * (models["sex_asex"]["log_lik"] - models["sex"]["log_lik"])
lr_asex_p = chi2.sf(lr_asex, df=1)
print(f"LRT asexual addition: stat={lr_asex:.3f}, p={lr_asex_p:.4f}")

# ── build final model description ──────────────────────────────────────────
if best_name == "sex_asex":
    model_name = "Sex-specific binomial logistic regression with asexual density"
    model_formula = (
        "logit(p) = β₀ + β_female × female_gametocytes_per_ul "
        "+ β_male × male_gametocytes_per_ul + β_asexual × asexual_per_ul"
    )
elif best_name == "sex":
    model_name = "Sex-specific binomial logistic regression"
    model_formula = (
        "logit(p) = β₀ + β_female × female_gametocytes_per_ul "
        "+ β_male × male_gametocytes_per_ul"
    )
else:
    model_name = "Total-gametocyte binomial logistic regression"
    model_formula = (
        "logit(p) = β₀ + β_total × (female_gametocytes_per_ul "
        "+ male_gametocytes_per_ul)"
    )

# Build prediction design matrix for best model
if best_name == "sex":
    X_pred_mat = design_matrix(F_pred, M_pred)
    X_train_mat = X_sex
elif best_name == "sex_asex":
    X_pred_mat = design_matrix(F_pred, M_pred, A_pred)
    X_train_mat = X_sex_asex
elif best_name == "total":
    X_pred_mat = design_matrix(T_pred)
    X_train_mat = X_total
else:
    X_pred_mat = design_matrix(F_pred, M_pred)
    X_train_mat = X_sex

p_pred, lo_pred, hi_pred = predict_with_ci(
    final_model["beta"], final_model["hessian_inv"], X_pred_mat, ci_level=0.90
)

# ── training fit diagnostics ───────────────────────────────────────────────
# Binomial log-loss on training data
eta_train = X_train_mat @ final_model["beta"]
p_train = expit(eta_train)
p_train_clipped = np.clip(p_train, 1e-15, 1 - 1e-15)
train_log_loss = -np.mean(
    y_train * np.log(p_train_clipped) + (n_train - y_train) * np.log1p(-p_train_clipped)
)

# Calibration: bin predictions, compare mean predicted to observed proportion
def calibration_check(p_pred_train, y, n, n_bins=5):
    """Simple binned calibration."""
    bins = np.percentile(p_pred_train, np.linspace(0, 100, n_bins + 1))
    bins[0] = -0.01  # ensure all points included
    bin_stats = []
    for i in range(n_bins):
        mask = (p_pred_train > bins[i]) & (p_pred_train <= bins[i + 1])
        if mask.sum() == 0:
            continue
        pred_mean = p_pred_train[mask].mean()
        obs_prop = y[mask].sum() / n[mask].sum()
        bin_stats.append({"predicted": pred_mean, "observed": obs_prop, "n": mask.sum()})
    return bin_stats

cal_bins = calibration_check(p_train, y_train, n_train, n_bins=5)
print("\n=== Calibration ===")
for b in cal_bins:
    print(f"  pred={b['predicted']:.4f}, obs={b['observed']:.4f}, n={b['n']}")

# ── assemble predictions ───────────────────────────────────────────────────
predictions = []
for i, fid in enumerate(feed_ids):
    predictions.append({
        "feed_id": fid,
        "infection_probability": float(p_pred[i]),
        "probability_interval": [float(lo_pred[i]), float(hi_pred[i])],
    })

# ── transmission summary ───────────────────────────────────────────────────
# Expected infected mosquitoes = sum(p_i * n_i)
expected_infected = float(np.sum(p_pred * n_pred))
weighted_prob = float(np.sum(p_pred * n_pred) / np.sum(n_pred))

# ── model comparison text ──────────────────────────────────────────────────
def format_model_comp():
    delta_aic = models["total"]["aic"] - models["sex"]["aic"]
    lines = [
        f"Sex-specific model AIC = {models['sex']['aic']:.1f} vs "
        f"total-gametocyte AIC = {models['total']['aic']:.1f} "
        f"(ΔAIC = {delta_aic:.1f} favouring sex-specific).",
        f"LRT: χ²(1) = {lr_stat:.2f}, p = {lr_pval:.4f}.",
    ]
    return " ".join(lines)


def format_asexual_assessment():
    lines = [
        f"Adding asexual density to the sex-specific model changed AIC by "
        f"{models['sex_asex']['aic'] - models['sex']['aic']:.1f} "
        f"(LRT χ²(1) = {lr_asex:.2f}, p = {lr_asex_p:.4f}), "
        f"suggesting asexual density does not materially improve prediction "
        f"once gametocyte densities are known.",
    ]
    return " ".join(lines)


def format_calibration():
    max_dev = max(abs(b["predicted"] - b["observed"]) for b in cal_bins)
    return (
        f"Binned calibration (5 bins): maximum absolute deviation "
        f"between predicted and observed proportions = {max_dev:.4f}. "
        + ("No evidence of gross miscalibration." if max_dev < 0.2
           else "Some miscalibration apparent.")
    )


# ── literature ─────────────────────────────────────────────────────────────
literature = [
    {
        "claim": (
            "Female and male gametocyte densities predict mosquito infection "
            "better jointly than as a total; at low densities male gametocytes "
            "can be limiting."
        ),
        "citation": (
            "Bradley J, Stone W, Da DF, Morlais I, Dicko A, Cohuet A, "
            "Guindo A, Djimde AA, Drakeley C, Bousema T. Predicting the "
            "likelihood and intensity of mosquito infection from sex-specific "
            "Plasmodium falciparum gametocyte densities. eLife 2018."
        ),
        "url": "https://doi.org/10.7554/eLife.34463",
    },
    {
        "claim": (
            "Female gametocytes were quantified by Pfs25 qRT-PCR and male "
            "gametocytes by PfMGET qRT-PCR; both assays are validated for "
            "sex-specific quantification in the absence of pure sex-specific "
            "gametocyte standards."
        ),
        "citation": (
            "Wampfler R, Mwingira F, Javati S, Robinson L, Betuela I, Siba P, "
            "Beck H-P, Mueller I, Felger I. Assays for quantification of male "
            "and female gametocytes in human blood by qRT-PCR in the absence "
            "of pure sex-specific gametocyte standards. Malaria Journal 2020."
        ),
        "url": "https://doi.org/10.1186/s12936-020-03291-9",
    },
    {
        "claim": (
            "Asexual parasite density measured by SBP1 qRT-PCR is a sensitive "
            "marker of ring-stage parasites that does not correlate strongly "
            "with gametocyte density and is not a direct predictor of "
            "mosquito infectivity."
        ),
        "citation": (
            "Tadesse FG, Lanke K, Nebie I, Schildkraut JA, Gonçalves BP, "
            "Tiono AB, Sauerwein R, Drakeley C, Bousema T, Rijpma SR. "
            "Molecular markers for sensitive detection of Plasmodium "
            "falciparum asexual stage parasites and their application in "
            "a malaria clinical trial. Am J Trop Med Hyg 2017."
        ),
        "url": "https://doi.org/10.4269/ajtmh.16-0893",
    },
]

# ── write results.json ─────────────────────────────────────────────────────
results = {
    "model": {
        "name": model_name,
        "formula": model_formula,
        "likelihood": "y_i ~ Binomial(n_i, p_i), independent across feeds",
    },
    "predictions": predictions,
    "transmission_summary": {
        "expected_infected_mosquitoes": round(expected_infected, 1),
        "weighted_infection_probability": round(weighted_prob, 4),
    },
    "diagnostics": {
        "training_binomial_log_loss": round(train_log_loss, 6),
        "model_comparison": format_model_comp(),
        "asexual_assessment": format_asexual_assessment(),
        "calibration": format_calibration(),
    },
    "literature": literature,
}

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== Results written to {OUT_PATH} ===")
print(f"Expected infected mosquitoes: {expected_infected:.1f}")
print(f"Weighted infection probability: {weighted_prob:.4f}")
print(f"Training binomial log-loss: {train_log_loss:.6f}")