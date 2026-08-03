"""Deterministic half of the SELF-C07 score."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TARGETS = {
    10.0: 95.18592502,
    14.0: 610.32104612,
    21.0: 772.96746870,
    24.0: 588.23720271,
}
BASELINE_LOG10_RMSE = 0.470149
REFERENCE_LOG10_RMSE = 0.055201


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def grade(result: dict) -> dict:
    model = result.get("model", {})
    parameters = result.get("parameters", {})
    diagnostics = result.get("diagnostics", {})

    model_ok = all(
        isinstance(model.get(key), str) and bool(model[key].strip())
        for key in (
            "name",
            "equations",
            "driver_interpolation",
            "solver",
            "observation_model",
        )
    )
    parameter_keys = (
        "effective_production_per_day",
        "mean_maturation_days",
        "mature_clearance_rate_per_day",
        "mature_half_life_days",
    )
    parameters_ok = all(
        finite_number(parameters.get(key)) and parameters[key] > 0
        for key in parameter_keys
    )
    diagnostics_ok = finite_number(diagnostics.get("training_log10_rmse")) and all(
        isinstance(diagnostics.get(key), str) and bool(diagnostics[key].strip())
        for key in ("alternative_model", "identifiability")
    )

    submitted = {}
    for row in result.get("predictions", []):
        time = row.get("time_days")
        value = row.get("gametocytes_per_ml")
        if finite_number(time) and finite_number(value) and value > 0:
            submitted[float(time)] = float(value)
    predictions_ok = set(submitted) == set(TARGETS)

    schema_points = 2.5 * sum(
        (model_ok, parameters_ok, diagnostics_ok, predictions_ok)
    )

    prediction_points = 0.0
    log10_rmse = None
    if predictions_ok:
        log10_rmse = math.sqrt(
            sum(
                (math.log10(submitted[time]) - math.log10(target)) ** 2
                for time, target in TARGETS.items()
            )
            / len(TARGETS)
        )
        skill = (BASELINE_LOG10_RMSE - log10_rmse) / (
            BASELINE_LOG10_RMSE - REFERENCE_LOG10_RMSE
        )
        prediction_points = 25.0 * min(1.0, max(0.0, skill))

    half_life_points = 0.0
    if parameters_ok:
        implied = math.log(2) / parameters["mature_clearance_rate_per_day"]
        if abs(parameters["mature_half_life_days"] - implied) / implied <= 0.05:
            half_life_points = 10.0

    interval = parameters.get("mature_half_life_interval")
    interval_ok = (
        parameters_ok
        and isinstance(interval, list)
        and len(interval) == 2
        and all(finite_number(value) for value in interval)
        and 0
        < interval[0]
        <= parameters.get("mature_half_life_days", -1)
        <= interval[1]
    )
    interval_points = 5.0 if interval_ok else 0.0

    components = {
        "schema": round(schema_points, 3),
        "held_out_prediction": round(prediction_points, 3),
        "half_life_consistency": half_life_points,
        "uncertainty_interval": interval_points,
    }
    return {
        "score": round(sum(components.values()), 3),
        "maximum": 50,
        "components": components,
        "metrics": {"held_out_log10_rmse": log10_rmse},
        "checks": {
            "model": model_ok,
            "parameters": parameters_ok,
            "diagnostics": diagnostics_ok,
            "prediction_rows": predictions_ok,
            "half_life_interval": interval_ok,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    print(json.dumps(grade(result), indent=2))


if __name__ == "__main__":
    main()
