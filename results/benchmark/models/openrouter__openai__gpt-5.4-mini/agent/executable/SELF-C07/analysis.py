import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"

DETECTION_LIMIT = 2.0


def load_data():
    driver = pd.read_csv(INPUT / "asexual_driver.csv")
    gam = pd.read_csv(INPUT / "gametocytes.csv")
    preds = pd.read_csv(INPUT / "prediction_times.csv")
    return driver, gam, preds


@dataclass
class ModelSpec:
    n_immature: int
    name: str


class GametocyteModel:
    def __init__(self, driver_df, n_immature=3):
        self.n = n_immature
        t = driver_df["time_days"].to_numpy(float)
        x = driver_df["asexual_parasites_per_ml"].to_numpy(float)
        self.t_min = float(t.min())
        self.t_max = float(t.max())
        # Piecewise-linear interpolation on the supplied arithmetic-scale driver.
        self.driver = interp1d(t, x, kind="linear", bounds_error=False,
                               fill_value=(x[0], x[-1]), assume_sorted=True)

    def rhs(self, t, y, beta, k, c):
        A = float(self.driver(t))
        dydt = np.zeros_like(y)
        dydt[0] = beta * A - self.n * k * y[0]
        for i in range(1, self.n):
            dydt[i] = self.n * k * (y[i - 1] - y[i])
        dydt[-1] = self.n * k * y[self.n - 1] - c * y[-1]
        return dydt

    def predict(self, times, beta, tau, c):
        k = 1.0 / tau
        y0 = np.zeros(self.n + 1)
        t_eval = np.asarray(times, float)
        sol = solve_ivp(
            lambda t, y: self.rhs(t, y, beta, k, c),
            (float(t_eval.min()), float(t_eval.max())),
            y0,
            t_eval=t_eval,
            method="LSODA",
            rtol=1e-7,
            atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return np.clip(sol.y[-1], 1e-12, np.inf)


def neg_loglike(log_params, model, gam_df):
    beta, tau, c, sigma = np.exp(log_params)
    pred = model.predict(gam_df["time_days"].to_numpy(float), beta, tau, c)
    obs = gam_df["gametocytes_per_ml"].to_numpy(object)
    status = gam_df["status"].to_numpy(str)
    ll = 0.0
    for y, mu, st in zip(obs, pred, status):
        if st == "below_detection_limit":
            z = (np.log(DETECTION_LIMIT) - np.log(mu)) / sigma
            ll += np.log(norm.cdf(z) + 1e-300)
        else:
            y = float(y)
            ll += -np.log(y * sigma * np.sqrt(2 * np.pi)) - ((np.log(y) - np.log(mu)) ** 2) / (2 * sigma**2)
    return -ll


def fit_model(model, gam_df, starts):
    bounds = [(-20, -2), (np.log(1.0), np.log(20.0)), (np.log(1e-4), np.log(1.0)), (np.log(0.05), np.log(2.0))]
    best = None
    for x0 in starts:
        try:
            res = minimize(neg_loglike, x0=np.log(x0), args=(model, gam_df), method="L-BFGS-B", bounds=bounds)
        except Exception:
            continue
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("all starts failed")
    return best


def hessian_diag_from_inv(res):
    Hinv = np.asarray(res.hess_inv.todense()) if hasattr(res.hess_inv, "todense") else np.asarray(res.hess_inv)
    return np.diag(Hinv)


def summarize_fit(model, res, gam_df):
    params = np.exp(res.x)
    beta, tau, c, sigma = params
    pred = model.predict(gam_df["time_days"].to_numpy(float), beta, tau, c)
    y = gam_df.loc[gam_df["status"] == "detected", "gametocytes_per_ml"].astype(float).to_numpy()
    mu = pred[gam_df["status"].to_numpy() == "detected"]
    rmse = np.sqrt(np.mean((np.log10(y) - np.log10(mu)) ** 2))
    diag = hessian_diag_from_inv(res)
    se_logc = np.sqrt(max(diag[2], 0.0))
    hl = np.log(2.0) / c
    hl_lo = np.log(2.0) / np.exp(np.log(c) + 1.96 * se_logc)
    hl_hi = np.log(2.0) / np.exp(np.log(c) - 1.96 * se_logc)
    return {
        "beta": beta,
        "tau": tau,
        "c": c,
        "sigma": sigma,
        "hl": hl,
        "hl_ci": [hl_lo, hl_hi],
        "rmse": rmse,
        "loglike": -res.fun,
        "aic": 2 * 4 - 2 * (-res.fun),
    }


def main():
    driver, gam, preds = load_data()
    model3 = GametocyteModel(driver, n_immature=3)
    model1 = GametocyteModel(driver, n_immature=1)

    starts = [
        np.array([1e-6, 8.0, 0.1, 0.4]),
        np.array([1e-7, 6.0, 0.2, 0.5]),
        np.array([1e-5, 10.0, 0.05, 0.3]),
        np.array([1e-6, 12.0, 0.2, 0.2]),
    ]
    fit3 = fit_model(model3, gam, starts)
    sum3 = summarize_fit(model3, fit3, gam)

    fit1 = fit_model(model1, gam, starts)
    sum1 = summarize_fit(model1, fit1, gam)

    beta, tau, c, sigma = np.exp(fit3.x)
    pred_times = preds["time_days"].to_numpy(float)
    pred_vals = model3.predict(pred_times, beta, tau, c)

    traj_t = np.round(np.arange(driver["time_days"].min(), driver["time_days"].max() + 1e-9, 0.1), 10)
    traj_vals = model3.predict(traj_t, beta, tau, c)

    pd.DataFrame({"time_days": traj_t, "mature_gametocytes_per_ml": traj_vals}).to_csv(ROOT / "trajectory.csv", index=False)

    report = {
        "model": {
            "name": "3-immature-compartment Erlang maturation model",
            "equations": "dI1/dt = beta*A(t) - 3k I1; dIi/dt = 3k(I{i-1}-Ii) for i=2..3; dM/dt = 3k I3 - cM, with A(t) the fixed asexual driver in parasites/mL and M in gametocytes/mL.",
            "driver_interpolation": "piecewise-linear interpolation on the supplied arithmetic-scale asexual trajectory; values outside the range are held constant at the boundary values.",
            "solver": "SciPy solve_ivp with LSODA, rtol=1e-7, atol=1e-10",
            "observation_model": "lognormal observation error for detected values; ND is left-censored below 2 gametocytes/mL via the lognormal CDF"
        },
        "parameters": {
            "effective_production_per_day": float(beta),
            "mean_maturation_days": float(tau),
            "mature_clearance_rate_per_day": float(c),
            "mature_half_life_days": float(sum3["hl"]),
            "mature_half_life_interval": [float(sum3["hl_ci"][0]), float(sum3["hl_ci"][1])]
        },
        "predictions": [
            {"time_days": float(t), "gametocytes_per_ml": float(v)} for t, v in zip(pred_times, pred_vals)
        ],
        "diagnostics": {
            "training_log10_rmse": float(sum3["rmse"]),
            "alternative_model": f"1-immature-compartment model had log-likelihood {sum1['loglike']:.2f} and AIC {sum1['aic']:.2f} versus {sum3['loglike']:.2f} and {sum3['aic']:.2f} for the 3-compartment model.",
            "identifiability": "beta, maturation time, and clearance are partially confounded by the short profile and the single ND value; the mature half-life is better identified than the immature-chain shape."
        }
    }

    with open(ROOT / "results.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
