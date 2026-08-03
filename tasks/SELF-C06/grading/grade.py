"""Deterministic half of the SELF-C06 score."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TARGETS = {
    7.5: 13.66,
    9.5: 98.17,
    11.5: 1105.13,
    12.0: 1995.24,
}
BASELINE_LOG10_RMSE = 0.902027
REFERENCE_LOG10_RMSE = 0.220527


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def grade(result: dict) -> dict:
    model = result.get("model", {})
    growth = result.get("growth", {})
    diagnostics = result.get("diagnostics", {})
    predictions = result.get("predictions", [])

    model_ok = all(
        isinstance(model.get(key), str) and bool(model[key].strip())
        for key in ("name", "formula", "observation_model")
    )
    diagnostics_ok = finite_number(diagnostics.get("training_log10_rmse")) and (
        isinstance(diagnostics.get("baseline_comparison"), str)
        and bool(diagnostics["baseline_comparison"].strip())
    )
    rate = growth.get("log10_rate_per_day")
    pmf = growth.get("pmf_48h")
    growth_ok = finite_number(rate) and finite_number(pmf) and pmf > 0

    submitted = {}
    for row in predictions:
        time = row.get("time_days")
        value = row.get("parasites_per_ml")
        if finite_number(time) and finite_number(value) and value > 0:
            submitted[float(time)] = float(value)
    predictions_ok = set(submitted) == set(TARGETS)

    schema_checks = [model_ok, diagnostics_ok, growth_ok, predictions_ok]
    schema_points = 2.5 * sum(schema_checks)

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
        skill = min(1.0, max(0.0, skill))
        prediction_points = 25.0 * skill

    pmf_points = 0.0
    if growth_ok:
        implied_pmf = 10 ** (2 * rate)
        if abs(pmf - implied_pmf) / implied_pmf <= 0.05:
            pmf_points = 10.0

    interval = growth.get("pmf_48h_interval")
    interval_ok = (
        isinstance(interval, list)
        and len(interval) == 2
        and all(finite_number(value) for value in interval)
        and 0 < interval[0] <= pmf <= interval[1]
    )
    interval_points = 5.0 if interval_ok else 0.0

    components = {
        "schema": round(schema_points, 3),
        "held_out_prediction": round(prediction_points, 3),
        "pmf_consistency": pmf_points,
        "uncertainty_interval": interval_points,
    }
    return {
        "score": round(sum(components.values()), 3),
        "maximum": 50,
        "components": components,
        "metrics": {"held_out_log10_rmse": log10_rmse},
        "checks": {
            "model": model_ok,
            "diagnostics": diagnostics_ok,
            "growth": growth_ok,
            "prediction_rows": predictions_ok,
            "pmf_interval": interval_ok,
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
