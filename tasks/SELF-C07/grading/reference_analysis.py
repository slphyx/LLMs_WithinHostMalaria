"""Reproduce the SELF-C07 reference fit."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import log_ndtr


ROOT = Path(__file__).parents[1]
SIGMA = 0.08
LIMIT = 2.0


def read_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "input" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


driver_rows = read_rows("asexual_driver.csv")
DRIVER_TIME = np.array([float(row["time_days"]) for row in driver_rows])
DRIVER_VALUE = np.array(
    [float(row["asexual_parasites_per_ml"]) for row in driver_rows]
)
observations = read_rows("gametocytes.csv")
OBS_TIME = np.array([float(row["time_days"]) for row in observations])
OBS_VALUE = np.array(
    [
        np.nan if row["gametocytes_per_ml"] == "ND"
        else float(row["gametocytes_per_ml"])
        for row in observations
    ]
)


def solve(
    log_parameters: np.ndarray,
    times: np.ndarray,
    immature_compartments: int = 4,
) -> np.ndarray:
    production, maturation_days, half_life = np.exp(log_parameters)
    transition = immature_compartments / maturation_days
    clearance = math.log(2) / half_life

    def rhs(time: float, state: np.ndarray) -> list[float]:
        asexual = np.interp(time, DRIVER_TIME, DRIVER_VALUE)
        derivative = [
            production * asexual - (transition + 0.03) * state[0]
        ]
        derivative.extend(
            transition * state[index - 1]
            - (transition + 0.03) * state[index]
            for index in range(1, immature_compartments)
        )
        derivative.append(
            transition * state[immature_compartments - 1]
            - clearance * state[immature_compartments]
        )
        return derivative

    fit = solve_ivp(
        rhs,
        (0, float(max(times))),
        np.zeros(immature_compartments + 1),
        t_eval=times,
        rtol=1e-8,
        atol=1e-10,
    )
    return fit.y[-1]


def residuals(
    log_parameters: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    immature_compartments: int,
) -> np.ndarray:
    prediction = solve(log_parameters, times, immature_compartments)
    detected = ~np.isnan(values)
    residual = (
        np.log10(prediction[detected]) - np.log10(values[detected])
    ) / SIGMA
    censored = []
    if np.any(~detected):
        z = (
            math.log10(LIMIT) - math.log10(prediction[~detected][0])
        ) / SIGMA
        censored = [math.sqrt(max(0, -2 * float(log_ndtr(z))))]
    _, maturation_days, half_life = np.exp(log_parameters)
    return np.r_[
        residual,
        censored,
        (maturation_days - 8.39) / 1.03,
        (half_life - 5.1) / 0.51,
    ]


def fit_model(
    times: np.ndarray,
    values: np.ndarray,
    immature_compartments: int,
):
    return least_squares(
        residuals,
        np.log([0.02, 8.4, 5.1]),
        args=(times, values, immature_compartments),
        bounds=(np.log([1e-4, 3, 0.5]), np.log([1, 15, 100])),
        diff_step=1e-4,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )


fit = fit_model(OBS_TIME, OBS_VALUE, 4)
production, maturation_days, half_life = np.exp(fit.x)
covariance = np.linalg.inv(fit.jac.T @ fit.jac)
half_life_se = math.sqrt(covariance[2, 2])
half_life_interval = np.exp(
    [fit.x[2] - 1.96 * half_life_se, fit.x[2] + 1.96 * half_life_se]
)
prediction_times = np.array(
    [float(row["time_days"]) for row in read_rows("prediction_times.csv")]
)
predictions = solve(fit.x, prediction_times)
fitted = solve(fit.x, OBS_TIME)
detected = ~np.isnan(OBS_VALUE)
training_rmse = math.sqrt(
    np.mean(
        (
            np.log10(fitted[detected])
            - np.log10(OBS_VALUE[detected])
        )
        ** 2
    )
)

validation = np.isin(OBS_TIME, [13.0, 18.0])
validation_rmse = {}
for compartments in (1, 4):
    validation_fit = fit_model(
        OBS_TIME[~validation],
        OBS_VALUE[~validation],
        compartments,
    )
    validation_prediction = solve(
        validation_fit.x,
        OBS_TIME[validation],
        compartments,
    )
    validation_rmse[compartments] = math.sqrt(
        np.mean(
            (
                np.log10(validation_prediction)
                - np.log10(OBS_VALUE[validation])
            )
            ** 2
        )
    )

result = {
    "model": {
        "name": "four-stage immature transit model",
        "equations": (
            "dG1/dt=qA-(k+mu_i)G1; dGj/dt=kG(j-1)-(k+mu_i)Gj; "
            "dG5/dt=kG4-mu_mG5"
        ),
        "driver_interpolation": "piecewise linear",
        "solver": "SciPy solve_ivp RK45, rtol=1e-8, atol=1e-10",
        "observation_model": (
            "Gaussian error on log10 density; ND contributes the normal CDF "
            "at 2 gametocytes/mL"
        ),
    },
    "parameters": {
        "effective_production_per_day": production,
        "mean_maturation_days": maturation_days,
        "mature_clearance_rate_per_day": math.log(2) / half_life,
        "mature_half_life_days": half_life,
        "mature_half_life_interval": half_life_interval.tolist(),
    },
    "predictions": [
        {
            "time_days": float(time),
            "gametocytes_per_ml": float(value),
        }
        for time, value in zip(prediction_times, predictions)
    ],
    "diagnostics": {
        "training_log10_rmse": training_rmse,
        "alternative_model": (
            "With days 13 and 18 withheld from fitting, log10 RMSE was "
            f"{validation_rmse[4]:.3f} for four immature stages and "
            f"{validation_rmse[1]:.3f} for one."
        ),
        "identifiability": (
            "Production, maturation and persistence are confounded; published "
            "maturation and half-life estimates were used as explicit weak "
            "constraints."
        ),
    },
}
print(json.dumps(result, indent=2))
