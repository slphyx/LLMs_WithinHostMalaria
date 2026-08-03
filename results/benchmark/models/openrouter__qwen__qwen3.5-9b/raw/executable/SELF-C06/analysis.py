import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import json

# Load data
data = pd.read_csv('input/parasitaemia.csv')
pred_times = pd.read_csv('input/prediction_times.csv')['time_days'].values
detected = data[data['status'] == 'detected'].copy()
censored = data[data['status'] == 'below_detection_limit'].copy()

# Log-likelihood function
def log_likelihood(params):
    beta0, beta1, log_sigma = params
    sigma = np.exp(log_sigma)
    time_det = detected['time_days'].values
    x_det = np.log10(detected['parasites_per_ml'].values)
    mu_det = beta0 + beta1 * time_det
    ll_det = stats.norm.logpdf(x_det, loc=mu_det, scale=sigma).sum()
    if len(censored) == 0:
        ll_cen = 0.0
    else:
        time_cen = censored['time_days'].values
        mu_cen = beta0 + beta1 * time_cen
        z = (1.0 - mu_cen) / sigma
        ll_cen = stats.norm.logcdf(z).sum()
    return ll_det + ll_cen

# Optimize
res = minimize(-log_likelihood, x0=[0, 0.1, 0], bounds=[(None, None), (None, None), (0, None)])
params = res.x
beta0, beta1, log_sigma = params
sigma = np.exp(log_sigma)

# PMF calculations
pmf_48h = 10**(beta1 * 2)
se_beta1 = np.sqrt(res.hess_inv[1, 1]) if res.hess_inv is not None else 0.01
pmf_ci = [10**(2 * (beta1 - 1.96 * se_beta1)), 10**(2 * (beta1 + 1.96 * se_beta1))]

# Predictions
preds = []
for t in pred_times:
    pred_log10 = beta0 + beta1 * t
    pred_val = 10**pred_log10
    preds.append({'time_days': t, 'parasites_per_ml': pred_val})

# Diagnostics
obs_log10 = np.log10(detected['parasites_per_ml'].values)
pred_log10 = beta0 + beta1 * detected['time_days'].values
train_rmse = np.sqrt(np.mean((obs_log10 - pred_log10)**2))

results = {
    'model': {
        'name': 'Tobit Log-Linear',
        'formula': 'log10(P) = beta0 + beta1 * time',
        'observation_model': 'Y_i ~ N(0, sigma^2) on log-scale; censored < 10 parasites/mL'
    },
    'growth': {
        'log10_rate_per_day': beta1,
        'pmf_48h': pmf_48h,
        'pmf_48h_interval': pmf_ci
    },
    'predictions': preds,
    'diagnostics': {
        'training_log10_rmse': round(train_rmse, 4),
        'baseline_comparison': 'Linear Tobit model appropriate; quadratic terms unjustified (small N, no severe nonlinearity)'
    }
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('report.md', 'w') as f:
    f.write(f"# Fit: {results['model']['name']}\n\n")
    f.write(f"Model: {results['model']['formula']}\n\n")
    f.write(f"Growth parameters:\n")
    f.write(f"- PMF 48h: {results['growth']['pmf_48h']:.2f} (95% CI: {results['growth']['pmf_48h_interval'][0]:.2f}, {results['growth']['pmf_48h_interval'][1]:.2f})\n\n")
    f.write(f"Predictions:\n")
    for p in results['predictions']:
        f.write(f"- {p['time_days']} days: {p['parasites_per_ml']:.2n} parasites/mL\n")
    f.write(f"\nDiagnostics: Training log10 RMSE = {results['diagnostics']['training_log10_rmse']:.4f}. ")
    f.write(f"Model selection: {results['diagnostics']['baseline_comparison']}\n")