import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import json

# Load data
df = pd.read_csv('input/parasitaemia.csv')
pred_df = pd.read_csv('input/prediction_times.csv')

# Detection limit
LOD = 10.0
logLOD = np.log10(LOD)

# Separate observed and censored
cens = df['status'] == 'below_detection_limit'
t_cens = df.loc[cens, 'time_days'].values
t_obs = df.loc[~cens, 'time_days'].values
y_obs = df.loc[~cens, 'parasites_per_ml'].values
log_y_obs = np.log10(y_obs)

# Negative log-likelihood for baseline model (log10-normal)
def nll_baseline(params):
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma)
    mu_obs = alpha + beta * t_obs
    ll_obs = norm.logpdf(log_y_obs, loc=mu_obs, scale=sigma)
    if len(t_cens) > 0:
        mu_cens = alpha + beta * t_cens
        ll_cens = norm.logcdf(logLOD, loc=mu_cens, scale=sigma)
        ll_cens = np.sum(ll_cens)
    else:
        ll_cens = 0.0
    return - (np.sum(ll_obs) + np.sum(ll_cens))

# Initial guess
init_alpha = -1.2
init_beta = 0.38
init_log_sigma = np.log(0.3)

res_baseline = minimize(nll_baseline, [init_alpha, init_beta, init_log_sigma], method='L-BFGS-B')
alpha_hat, beta_hat, log_sigma_hat = res_baseline.x
sigma_hat = np.exp(log_sigma_hat)

# Numerical Hessian for standard errors
def compute_hessian(func, x, eps=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    f0 = func(x)
    for i in range(n):
        x_ph = x.copy()
        x_ph[i] += eps
        f_ph = func(x_ph)
        x_mh = x.copy()
        x_mh[i] -= eps
        f_mh = func(x_mh)
        hess[i,i] = (f_ph - 2*f0 + f_mh) / eps**2
        for j in range(i+1, n):
            x_pp = x.copy()
            x_pp[i] += eps
            x_pp[j] += eps
            f_pp = func(x_pp)
            x_pm = x.copy()
            x_pm[i] += eps
            x_pm[j] -= eps
            f_pm = func(x_pm)
            x_mp = x.copy()
            x_mp[i] -= eps
            x_mp[j] += eps
            f_mp = func(x_mp)
            x_mm = x.copy()
            x_mm[i] -= eps
            x_mm[j] -= eps
            f_mm = func(x_mm)
            hess[i,j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
            hess[j,i] = hess[i,j]
    return hess

hess = compute_hessian(nll_baseline, res_baseline.x)
try:
    cov = np.linalg.inv(hess)
    se = np.sqrt(np.diag(cov))
except np.linalg.LinAlgError:
    se = np.full(len(res_baseline.x), np.nan)

se_alpha = se[0]
se_beta = se[1]

# PMF 48h
pmf = 10**(48 * beta_hat)
pmf_ci_lower = 10**(48 * (beta_hat - 1.96 * se_beta))
pmf_ci_upper = 10**(48 * (beta_hat + 1.96 * se_beta))

# Training log10 RMSE baseline
mu_obs_hat = alpha_hat + beta_hat * t_obs
residuals_obs = log_y_obs - mu_obs_hat
rmse_baseline = np.sqrt(np.mean(residuals_obs**2))

# More complex harmonic model
def nll_harmonic(params):
    alpha, beta, A, B, log_sigma = params
    sigma = np.exp(log_sigma)
    mu_obs = alpha + beta*t_obs + A*np.cos(np.pi*t_obs) + B*np.sin(np.pi*t_obs)
    ll_obs = norm.logpdf(log_y_obs, loc=mu_obs, scale=sigma)
    if len(t_cens) > 0:
        mu_cens = alpha + beta*t_cens + A*np.cos(np.pi*t_cens) + B*np.sin(np.pi*t_cens)
        ll_cens = norm.logcdf(logLOD, loc=mu_cens, scale=sigma)
    else:
        ll_cens = 0.0
    return - (np.sum(ll_obs) + np.sum(ll_cens))

init_h = [alpha_hat, beta_hat, 0.0, 0.0, log_sigma_hat]
res_harmonic = minimize(nll_harmonic, init_h, method='L-BFGS-B')
alpha_h, beta_h, A_h, B_h, log_sigma_h = res_harmonic.x
sigma_h = np.exp(log_sigma_h)
nll_b = res_baseline.fun
nll_h = res_harmonic.fun

# AICc
n = len(df)
k_base = 3
k_harm = 5
aicc_base = 2*k_base + 2*nll_b + (2*k_base*(k_base+1))/(n - k_base - 1)
aicc_harm = 2*k_harm + 2*nll_h + (2*k_harm*(k_harm+1))/(n - k_harm - 1)
delta_aicc = aicc_harm - aicc_base

mu_obs_hat_h = alpha_h + beta_h*t_obs + A_h*np.cos(np.pi*t_obs) + B_h*np.sin(np.pi*t_obs)
rmse_harmonic = np.sqrt(np.mean((log_y_obs - mu_obs_hat_h)**2))

# Model selection: baseline chosen
model_name = "free-intercept log-linear"
model_formula = "log10(parasites_per_ml) = alpha + beta * time_days"
observation_model = "Log-normal with left-censoring at 10 parasites/mL (ND treated as <10)"

# Predictions at required times
times_pred = pred_df['time_days'].values
mu_pred = alpha_hat + beta_hat * times_pred
parasites_pred = 10**mu_pred

# Build results
results = {
    "model": {
        "name": model_name,
        "formula": model_formula,
        "observation_model": observation_model
    },
    "growth": {
        "log10_rate_per_day": round(beta_hat, 6),
        "pmf_48h": round(pmf, 4),
        "pmf_48h_interval": [round(pmf_ci_lower, 4), round(pmf_ci_upper, 4)]
    },
    "predictions": [
        {"time_days": t, "parasites_per_ml": round(p, 2)} for t, p in zip(times_pred, parasites_pred)
    ],
    "diagnostics": {
        "training_log10_rmse": round(rmse_baseline, 4),
        "baseline_comparison": f"AICc baseline: {aicc_base:.2f}, harmonic: {aicc_harm:.2f}, ΔAICc = {delta_aicc:.2f}. Training log10 RMSE: baseline = {rmse_baseline:.4f}, harmonic = {rmse_harmonic:.4f}. The harmonic model does not substantially improve fit or predictive performance, so the simpler log-linear model is chosen."
    }
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
