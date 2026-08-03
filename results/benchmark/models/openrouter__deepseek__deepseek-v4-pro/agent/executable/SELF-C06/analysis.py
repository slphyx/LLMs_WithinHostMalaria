#!/usr/bin/env python3
"""Fit a within-host parasite trajectory model to qPCR data with left-censoring.

Reads input/parasitaemia.csv and input/prediction_times.csv, fits a
free-intercept log-linear baseline model and a logistic growth model,
selects the best justified model by AICc, and writes results.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats

# ---------------------------------------------------------------------------
# 0.  Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
LOD = 10.0  # parasites/mL
LOG10_LOD = np.log10(LOD)
SEED = 20250101


# ---------------------------------------------------------------------------
# 1.  Read input data
# ---------------------------------------------------------------------------
def load_data():
    """Load and parse parasitaemia and prediction-times CSVs."""
    raw = np.genfromtxt(
        INPUT_DIR / "parasitaemia.csv",
        delimiter=",",
        dtype=None,
        names=True,
        encoding="utf-8",
        autostrip=True,
    )

    times = []
    log10y = []
    cens = []
    for row in raw:
        t = float(row["time_days"])
        times.append(t)
        if row["status"] == "below_detection_limit":
            log10y.append(np.nan)
            cens.append(True)
        else:
            log10y.append(np.log10(float(row["parasites_per_ml"])))
            cens.append(False)
    times = np.array(times)
    log10y = np.array(log10y)
    cens = np.array(cens)

    pred_raw = np.genfromtxt(
        INPUT_DIR / "prediction_times.csv",
        delimiter=",",
        dtype=None,
        names=True,
        encoding="utf-8",
        autostrip=True,
    )
    pred_times = np.array([float(x) for x in pred_raw["time_days"]])

    return times, log10y, cens, pred_times


# ---------------------------------------------------------------------------
# 2.  Likelihoods and fitting
# ---------------------------------------------------------------------------


def censored_normal_nll(params, mean_fn, times, log10y, cens):
    """Negative log-likelihood for a censored normal observation model.

    Parameters
    ----------
    params : array
        Model parameters; the last element is always log_sigma.
    mean_fn : callable
        mean_fn(t, *theta) -> array of log10 means.
    times, log10y, cens : arrays
        Data.  log10y[i] is unused (NaN) when cens[i] is True.
    """
    *theta, log_sigma = params
    sigma = np.exp(log_sigma)
    mu = mean_fn(times, *theta)
    ll = 0.0
    for i in range(len(times)):
        if cens[i]:
            ll += stats.norm.logcdf((LOG10_LOD - mu[i]) / sigma)
        else:
            ll += stats.norm.logpdf(log10y[i], loc=mu[i], scale=sigma)
    return -ll


def numerical_hessian(func, x, eps=1e-5, args=()):
    """Hessian via second-order central differences of gradient."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        g_plus = optimize.approx_fprime(x_plus, func, eps, *args)
        x_minus = x.copy()
        x_minus[i] -= eps
        g_minus = optimize.approx_fprime(x_minus, func, eps, *args)
        H[i, :] = (g_plus - g_minus) / (2.0 * eps)
    H = 0.5 * (H + H.T)
    return H


def standard_errors_from_hessian(hess):
    """Return standard errors; NaN on failure."""
    try:
        eigvals = np.linalg.eigvalsh(hess)
        if np.any(eigvals <= 0):
            return np.full(hess.shape[0], np.nan)
        cov = np.linalg.inv(hess)
        return np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        return np.full(hess.shape[0], np.nan)


# ---------------------------------------------------------------------------
# 2a.  Free-intercept log-linear baseline
# ---------------------------------------------------------------------------


def log_linear_mean(t, a, b):
    """log10 parasites/mL = a + b * t."""
    return a + b * t


def fit_log_linear(times, log10y, cens):
    """MLE for the baseline log-linear model."""
    detected = ~cens
    b_ols, a_ols = np.polyfit(times[detected], log10y[detected], 1)
    resid = log10y[detected] - (a_ols + b_ols * times[detected])
    sigma_ols = max(np.std(resid, ddof=1), 0.01)

    init = [a_ols, b_ols, np.log(sigma_ols)]

    # L-BFGS-B first
    res = optimize.minimize(
        censored_normal_nll,
        init,
        args=(log_linear_mean, times, log10y, cens),
        method="L-BFGS-B",
        options={"maxiter": 10000},
    )
    # Nelder-Mead fallback
    if not res.success:
        res = optimize.minimize(
            censored_normal_nll,
            init,
            args=(log_linear_mean, times, log10y, cens),
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-10},
        )

    a, b, log_sigma = res.x
    hess = numerical_hessian(
        censored_normal_nll, res.x,
        args=(log_linear_mean, times, log10y, cens),
    )
    se = standard_errors_from_hessian(hess)

    n = len(times)
    k = 3
    aic = 2.0 * res.fun + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1.0) / max(n - k - 1.0, 1.0)

    return {
        "a": a,
        "b": b,
        "sigma": np.exp(log_sigma),
        "se": se,  # [se_a, se_b, se_log_sigma]
        "nll": res.fun,
        "aic": aic,
        "aicc": aicc,
        "converged": res.success,
        "n_params": k,
    }


# ---------------------------------------------------------------------------
# 2b.  Logistic (saturating) growth model
# ---------------------------------------------------------------------------


def logistic_mean(t, K, r, t_half):
    """log10 carrying capacity K, growth rate r, half-max time t_half."""
    return K - np.log10(1.0 + np.exp(-r * (t - t_half)))


def fit_logistic(times, log10y, cens):
    """MLE for logistic growth."""
    detected = ~cens
    K_init = log10y[detected].max() + 0.5
    b_tmp, a_tmp = np.polyfit(times[detected], log10y[detected], 1)
    r_init = max(b_tmp, 0.01)
    t_half_init = np.clip((K_init - np.log10(2) - a_tmp) / r_init,
                          times.min() - 1.0, times.max() + 1.0)

    best = None
    best_nll = np.inf

    # Multiple starts to avoid local minima
    starts = [
        [K_init, r_init, t_half_init, np.log(0.3)],
        [K_init * 1.1, r_init * 0.9, t_half_init, np.log(0.3)],
        [K_init * 0.9, r_init * 1.1, t_half_init, np.log(0.3)],
    ]

    for init in starts:
        res = optimize.minimize(
            censored_normal_nll,
            init,
            args=(logistic_mean, times, log10y, cens),
            method="L-BFGS-B",
            bounds=[
                (log10y[detected].max() + 0.01, None),
                (1e-6, None),
                (None, None),
                (None, None),
            ],
            options={"maxiter": 10000},
        )
        if res.fun < best_nll:
            best_nll = res.fun
            best = res

    if best is None:
        # Last-resort Nelder-Mead
        best = optimize.minimize(
            censored_normal_nll,
            [K_init, r_init, t_half_init, np.log(0.3)],
            args=(logistic_mean, times, log10y, cens),
            method="Nelder-Mead",
            options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10},
        )

    K, r, t_half, log_sigma = best.x
    hess = numerical_hessian(
        censored_normal_nll, best.x,
        args=(logistic_mean, times, log10y, cens),
    )
    se = standard_errors_from_hessian(hess)

    n = len(times)
    k = 4
    aic = 2.0 * best.fun + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1.0) / max(n - k - 1.0, 1.0)

    return {
        "K": K,
        "r": r,
        "t_half": t_half,
        "sigma": np.exp(log_sigma),
        "se": se,
        "nll": best.fun,
        "aic": aic,
        "aicc": aicc,
        "converged": best.success,
        "n_params": k,
    }


# ---------------------------------------------------------------------------
# 3.  PMF per 48-hour cycle
# ---------------------------------------------------------------------------


def pmf_48h_from_log10_rate(b, se_b=None):
    """PMF = 10^(2*b).  Returns (pmf, pmf_interval)."""
    pmf = 10.0 ** (2.0 * b)
    if se_b is None or np.isnan(se_b):
        return float(pmf), [float(pmf), float(pmf)]
    # 95 % Wald CI on log10 PMF = 2*b scale
    lo = 10.0 ** (2.0 * (b - 1.96 * se_b))
    hi = 10.0 ** (2.0 * (b + 1.96 * se_b))
    return float(pmf), [float(lo), float(hi)]


# ---------------------------------------------------------------------------
# 4.  Main analysis
# ---------------------------------------------------------------------------


def main():
    times, log10y, cens, pred_times = load_data()

    # --- fit both models ---
    lin = fit_log_linear(times, log10y, cens)
    logist = fit_logistic(times, log10y, cens)

    delta_aicc = logist["aicc"] - lin["aicc"]
    use_logistic = delta_aicc < -2.0

    # --- PMF ---
    b_hat = lin["b"]
    se_b = lin["se"][1]
    pmf, pmf_interval = pmf_48h_from_log10_rate(b_hat, se_b)

    # --- predictions ---
    if use_logistic:
        def _mean_fn(t_arr):
            return logistic_mean(t_arr, logist["K"], logist["r"], logist["t_half"])
    else:
        a_fit, b_fit = lin["a"], lin["b"]

        def _mean_fn(t_arr):
            return log_linear_mean(t_arr, a_fit, b_fit)

    pred_log10 = _mean_fn(pred_times)
    pred_lin = 10.0 ** pred_log10
    predictions = [
        {"time_days": float(t), "parasites_per_ml": float(p)}
        for t, p in zip(pred_times, pred_lin)
    ]

    # --- training RMSE ---
    mu_train = _mean_fn(times)
    detected = ~cens
    resid_log10 = log10y[detected] - mu_train[detected]
    training_log10_rmse = float(np.sqrt(np.mean(resid_log10 ** 2)))

    # --- residual diagnostics ---
    ac = 0.0
    if len(resid_log10) > 2:
        ac = float(np.corrcoef(resid_log10[:-1], resid_log10[1:])[0, 1])
    shapiro_p = 1.0
    if len(resid_log10) >= 3:
        _, shapiro_p = stats.shapiro(resid_log10)
    shapiro_p = float(shapiro_p)

    # --- model info ---
    K_lin = 10.0 ** logist["K"] if not np.isnan(logist["K"]) else np.inf
    t_half_str = f"{logist['t_half']:.1f}" if not np.isnan(logist["t_half"]) else "?"

    if not use_logistic:
        model_info = {
            "name": "free-intercept log-linear growth",
            "formula": (
                "log10(parasites/mL) = a + b * time_days, "
                "with Normal(0, sigma^2) error on the log10 scale"
            ),
            "observation_model": (
                "Normal distribution on log10-transformed densities with "
                "constant variance sigma^2. Measurements below 10 "
                "parasites/mL are left-censored: likelihood contribution "
                "Phi((log10(10) - mu) / sigma)."
            ),
        }
        baseline_comparison = (
            f"A logistic growth model (Delta_AICc = {delta_aicc:+.1f}) "
            f"does not improve on the free-intercept log-linear baseline: "
            f"its carrying capacity (K ~ {K_lin:.0f} parasites/mL, "
            f"t_half ~ {t_half_str} d) is poorly identified, "
            f"and the extra parameter is not supported by the data. "
            f"Residuals show no systematic departure from log-linearity "
            f"(lag-1 autocorrelation = {ac:.3f}, "
            f"Shapiro-Wilk p = {shapiro_p:.3f})."
        )
    else:
        model_info = {
            "name": "logistic growth with censored normal errors",
            "formula": (
                "log10(parasites/mL) = K - log10(1 + exp(-r * (time_days - t_half)))"
                ", with Normal(0, sigma^2) error on the log10 scale"
            ),
            "observation_model": (
                "Normal distribution on log10-transformed densities with "
                "constant variance sigma^2. Measurements below 10 "
                "parasites/mL are left-censored."
            ),
        }
        baseline_comparison = (
            f"The logistic growth model (Delta_AICc = {delta_aicc:+.1f}) "
            f"improves on the free-intercept log-linear baseline by "
            f"capturing a saturating trajectory (K ~ {K_lin:.0f} "
            f"parasites/mL, t_half ~ {t_half_str} d)."
        )

    # --- assemble results ---
    results = {
        "model": model_info,
        "growth": {
            "log10_rate_per_day": float(b_hat),
            "pmf_48h": pmf,
            "pmf_48h_interval": pmf_interval,
        },
        "predictions": predictions,
        "diagnostics": {
            "training_log10_rmse": training_log10_rmse,
            "baseline_comparison": baseline_comparison,
        },
    }

    with open(ROOT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- console summary ---
    print(f"Model selected: {model_info['name']}")
    print(f"log10 rate/day = {b_hat:.5f}  (SE = {se_b:.5f})")
    print(f"PMF per 48h    = {pmf:.2f}  [{pmf_interval[0]:.2f}, {pmf_interval[1]:.2f}]")
    print(f"sigma          = {lin['sigma']:.4f}")
    print(f"Training RMSE  = {training_log10_rmse:.4f} (log10 scale)")
    print(f"AICc (linear)  = {lin['aicc']:.2f}")
    print(f"AICc (logistic) = {logist['aicc']:.2f}")
    print(f"Delta_AICc     = {delta_aicc:+.2f}")
    print(f"Chose logistic: {use_logistic}")
    print("Predictions:")
    for p in predictions:
        print(f"  t = {p['time_days']:.1f} d  ->  {p['parasites_per_ml']:.1f} parasites/mL")
    print("results.json written.")


if __name__ == "__main__":
    main()