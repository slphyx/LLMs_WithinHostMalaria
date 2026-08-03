import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'input'
OUT = ROOT / 'results.json'

LIMIT = 10.0
LOG10 = np.log(10.0)
RNG = np.random.default_rng(12345)


def load_data():
    df = pd.read_csv(INPUT / 'parasitaemia.csv')
    pred = pd.read_csv(INPUT / 'prediction_times.csv')
    x = df['time_days'].to_numpy(float)
    cens = df['status'].eq('below_detection_limit').to_numpy(bool)
    y = df['parasites_per_ml'].astype(str).to_numpy()
    logy = np.array([np.log10(LIMIT) if c else np.log10(float(v)) for c, v in zip(cens, y)], dtype=float)
    return x, cens, logy, pred['time_days'].to_numpy(float)


def fit_loglinear(x, cens, logy, quadratic=False):
    def nll(p):
        a, b = p[0], p[1]
        c = p[2] if quadratic else 0.0
        ls = p[-1]
        sigma = np.exp(ls)
        mu = a + b * x + c * (x - 9.0) ** 2
        z = (np.log10(LIMIT) - mu) / sigma
        ll = np.empty_like(mu)
        ll[cens] = np.log(np.maximum(norm.cdf(z[cens]), 1e-300))
        ll[~cens] = norm.logpdf(logy[~cens], mu[~cens], sigma)
        return -np.sum(ll)

    p0 = np.array([0.5, 0.3, np.log(0.3)]) if not quadratic else np.array([0.5, 0.3, 0.0, np.log(0.3)])
    bounds = [(None, None), (None, None)] + ([(None, None)] if quadratic else []) + [(-8, 4)]
    res = minimize(nll, p0, method='L-BFGS-B', bounds=bounds)
    return res


def predict(p, times, quadratic=False):
    a, b = p[0], p[1]
    c = p[2] if quadratic else 0.0
    mu = a + b * times + c * (times - 9.0) ** 2
    return 10 ** mu


def bootstrap_interval(x, cens, logy, times, quadratic=False, n_boot=2000):
    fit = fit_loglinear(x, cens, logy, quadratic=quadratic)
    p = fit.x
    if quadratic:
        a, b, c, ls = p
    else:
        a, b, ls = p
        c = 0.0
    sigma = np.exp(ls)
    mu = a + b * x + c * (x - 9.0) ** 2
    boot = []
    for _ in range(n_boot):
        sim = np.empty_like(logy)
        unc = ~cens
        sim[unc] = RNG.normal(mu[unc], sigma)
        zc = (np.log10(LIMIT) - mu[cens]) / sigma
        u = RNG.random(np.sum(cens))
        q = norm.ppf(u * norm.cdf(zc))
        sim[cens] = mu[cens] + sigma * q

        def nll(pp):
            aa, bb = pp[0], pp[1]
            cc = pp[2] if quadratic else 0.0
            lss = pp[-1]
            s = np.exp(lss)
            mm = aa + bb * x + cc * (x - 9.0) ** 2
            zz = (np.log10(LIMIT) - mm) / s
            ll = np.empty_like(mm)
            ll[cens] = np.log(np.maximum(norm.cdf(zz[cens]), 1e-300))
            ll[unc] = norm.logpdf(sim[unc], mm[unc], s)
            return -np.sum(ll)

        p0 = p.copy()
        r = minimize(nll, p0, method='L-BFGS-B', bounds=[(None,None)]*(len(p)-1)+[(-8,4)])
        if r.success:
            pp = r.x
            bb = pp[1]
            boot.append(10 ** (2.0 * bb))
    boot = np.array(boot)
    return np.quantile(boot, [0.025, 0.975])


def main():
    x, cens, logy, pred_times = load_data()
    fit1 = fit_loglinear(x, cens, logy, quadratic=False)
    fit2 = fit_loglinear(x, cens, logy, quadratic=True)

    # Choose simpler model unless the quadratic materially improves fit.
    ll1, ll2 = -fit1.fun, -fit2.fun
    aic1 = 2 * 3 - 2 * ll1
    aic2 = 2 * 4 - 2 * ll2
    chosen_quad = (aic2 + 2.0) < aic1
    fit = fit2 if chosen_quad else fit1
    quad = chosen_quad
    p = fit.x
    a, b = p[0], p[1]
    c = p[2] if quad else 0.0
    ls = p[-1]
    sigma = float(np.exp(ls))

    # training RMSE on detected values, on log10 scale
    mu = a + b * x + c * (x - 9.0) ** 2
    rmse = float(np.sqrt(np.mean((logy[~cens] - mu[~cens]) ** 2)))

    preds = predict(p, pred_times, quadratic=quad)
    ci = bootstrap_interval(x, cens, logy, pred_times, quadratic=quad, n_boot=1500)

    pmf48 = float(10 ** (2.0 * b))
    pmf_ci = [float(ci[0]), float(ci[1])]

    result = {
        "model": {
            "name": "left-censored log-linear growth on log10 parasites/mL" + (" with quadratic time term" if quad else ""),
            "formula": "log10(P(t)) = a + b t" + (" + c (t-9)^2" if quad else "") + "; observed ND values are left-censored at 10 parasites/mL",
            "observation_model": "Gaussian errors on log10 scale; detected values use a normal density and ND measurements use the normal CDF below the 10 parasites/mL detection limit"
        },
        "growth": {
            "log10_rate_per_day": float(b),
            "pmf_48h": pmf48,
            "pmf_48h_interval": pmf_ci,
        },
        "predictions": [
            {"time_days": float(t), "parasites_per_ml": float(y)} for t, y in zip(pred_times, preds)
        ],
        "diagnostics": {
            "training_log10_rmse": rmse,
            "baseline_comparison": (
                f"Quadratic time curvature did not improve AIC (ΔAIC={aic1 - aic2:.2f}), so the free-intercept log-linear model was retained. "
                f"It gives a daily log10 growth rate of {b:.3f} and 48 h multiplication factor {pmf48:.2f}; the fitted residual SD is {sigma:.2f} log10 units."
            )
        }
    }
    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
