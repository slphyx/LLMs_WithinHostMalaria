"""Deterministic half of the SELF-C08 score."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TARGETS = {
    "F09": (86, 11),
    "F11": (95, 0),
    "F13": (90, 2),
    "F15": (113, 0),
    "F18": (76, 0),
    "F24": (120, 41),
    "F26": (72, 12),
    "F35": (92, 1),
    "F36": (79, 0),
}
BASELINE_LOG_LOSS = 0.285357177
REFERENCE_LOG_LOSS = 0.194267588


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def grade(result: dict) -> dict:
    model = result.get("model", {})
    diagnostics = result.get("diagnostics", {})
    summary = result.get("transmission_summary", {})
    predictions = result.get("predictions", [])
    literature = result.get("literature", [])

    model_ok = all(nonempty_text(model.get(key)) for key in ("name", "formula", "likelihood"))
    diagnostics_ok = finite_number(diagnostics.get("training_binomial_log_loss")) and all(
        nonempty_text(diagnostics.get(key))
        for key in ("model_comparison", "asexual_assessment", "calibration")
    )
    summary_ok = all(
        finite_number(summary.get(key))
        for key in ("expected_infected_mosquitoes", "weighted_infection_probability")
    )

    submitted = {}
    for row in predictions:
        feed_id = row.get("feed_id")
        probability = row.get("infection_probability")
        if (
            isinstance(feed_id, str)
            and finite_number(probability)
            and 0 < probability < 1
            and feed_id not in submitted
        ):
            submitted[feed_id] = (float(probability), row.get("probability_interval"))
    predictions_ok = len(predictions) == len(TARGETS) and set(submitted) == set(TARGETS)

    schema_points = 2.5 * sum((model_ok, diagnostics_ok, summary_ok, predictions_ok))

    hidden_log_loss = None
    prediction_points = 0.0
    if predictions_ok:
        total_mosquitoes = sum(examined for examined, _ in TARGETS.values())
        hidden_log_loss = -sum(
            infected * math.log(submitted[feed_id][0])
            + (examined - infected) * math.log1p(-submitted[feed_id][0])
            for feed_id, (examined, infected) in TARGETS.items()
        ) / total_mosquitoes
        skill = (BASELINE_LOG_LOSS - hidden_log_loss) / (
            BASELINE_LOG_LOSS - REFERENCE_LOG_LOSS
        )
        prediction_points = 25.0 * min(1.0, max(0.0, skill))

    intervals_ok = predictions_ok and all(
        isinstance(interval, list)
        and len(interval) == 2
        and all(finite_number(bound) for bound in interval)
        and 0 <= interval[0] <= probability <= interval[1] <= 1
        for probability, interval in submitted.values()
    )
    interval_points = 5.0 if intervals_ok else 0.0

    summary_consistent = False
    if predictions_ok and summary_ok:
        expected = sum(
            TARGETS[feed_id][0] * probability
            for feed_id, (probability, _) in submitted.items()
        )
        weighted = expected / sum(examined for examined, _ in TARGETS.values())
        summary_consistent = (
            abs(summary["expected_infected_mosquitoes"] - expected)
            <= max(0.05, 0.005 * expected)
            and abs(summary["weighted_infection_probability"] - weighted) <= 0.001
        )
    summary_points = 5.0 if summary_consistent else 0.0

    literature_ok = (
        isinstance(literature, list)
        and len(literature) >= 3
        and all(
            isinstance(item, dict)
            and nonempty_text(item.get("claim"))
            and nonempty_text(item.get("citation"))
            and isinstance(item.get("url"), str)
            and item["url"].startswith(("https://", "http://"))
            for item in literature
        )
    )
    literature_points = 5.0 if literature_ok else 0.0

    components = {
        "schema": round(schema_points, 3),
        "held_out_prediction": round(prediction_points, 3),
        "probability_intervals": interval_points,
        "transmission_summary": summary_points,
        "literature_records": literature_points,
    }
    return {
        "score": round(sum(components.values()), 3),
        "maximum": 50,
        "components": components,
        "metrics": {"held_out_binomial_log_loss": hidden_log_loss},
        "checks": {
            "model": model_ok,
            "diagnostics": diagnostics_ok,
            "summary": summary_ok,
            "prediction_rows": predictions_ok,
            "probability_intervals": intervals_ok,
            "summary_consistency": summary_consistent,
            "literature_records": literature_ok,
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
