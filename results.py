"""Readable exports from Inspect's canonical evaluation logs."""

from __future__ import annotations

import math
from pathlib import Path

from inspect_ai.log import EvalLog
from pydantic_core import to_json

from billing import billing_for_model, model_billing


def export_results(logs: list[EvalLog], run_dir: Path) -> dict:
    """Project canonical logs and build raw-versus-agent comparisons."""
    summary_rows = []
    for log in logs:
        model_name = str(log.eval.model)
        condition = log.eval.metadata.get("condition", "raw")
        model_dir = (
            run_dir
            / "models"
            / model_name.replace("/", "__")
            / condition
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for sample in log.samples or []:
            score = next(iter(sample.scores.values()), None) if sample.scores else None
            metadata = score.metadata if score and score.metadata else {}
            store = sample.store or {}
            output = sample.output.model_dump(mode="json") if sample.output else {}
            choice = output.get("choices", [{}])[0] if output.get("choices") else {}

            # Inspect keeps the full OpenRouter response in model events. Its
            # provider-reported cost is more accurate than estimating from tokens.
            model_cost, billed_calls = model_billing(sample)
            web_cost = float(store.get("web_search_cost", 0) or 0)

            judges = [
                {
                    **judge,
                    "billing": billing_for_model(
                        billed_calls, judge.get("judge")
                    ),
                }
                for judge in metadata.get("judges") or []
            ]
            failures = [
                {
                    **failure,
                    "billing": billing_for_model(
                        billed_calls, failure.get("judge")
                    ),
                }
                for failure in metadata.get("judge_failures") or []
            ]
            samples.append(
                {
                    "id": sample.id,
                    "epoch": sample.epoch,
                    "uuid": sample.uuid,
                    "condition": condition,
                    "question": sample.input,
                    "reference_answer": sample.target,
                    "source": sample.metadata.get("source"),
                    "section": sample.metadata.get("section"),
                    "difficulty": sample.metadata.get("difficulty"),
                    "answer_validation": sample.metadata.get("answer_validation"),
                    "response": {
                        "requested_model": model_name,
                        "served_model": output.get("model"),
                        "answer": output.get("completion"),
                        "stop_reason": choice.get("stop_reason"),
                        "usage": output.get("usage"),
                        "generation_time": output.get("time"),
                        "model_metadata": output.get("metadata"),
                        "fallback": output.get("fallback"),
                        "error": output.get("error"),
                        "billing": billing_for_model(
                            billed_calls, model_name
                        ),
                    },
                    "judgement": {
                        "score": score.value if score else None,
                        "scored": bool(
                            score
                            and not (
                                isinstance(score.value, float)
                                and math.isnan(score.value)
                            )
                        ),
                        "explanation": score.explanation if score else None,
                        "criteria": metadata.get("criteria"),
                        "error_assessment": metadata.get("error_assessment"),
                        "judges": judges,
                        "failures": failures,
                    },
                    "execution": {
                        "billing": {
                            "source": "openrouter_response",
                            "model_cost": model_cost,
                            "web_search_cost": web_cost,
                            "total_cost": model_cost + web_cost,
                            "calls": len(billed_calls),
                            "model_calls": billed_calls,
                        },
                        "web_search_calls": store.get("web_search_calls", []),
                        "model_usage": {
                            name: usage.model_dump(mode="json")
                            for name, usage in sample.model_usage.items()
                        },
                        "started_at": sample.started_at,
                        "completed_at": sample.completed_at,
                        "total_time": sample.total_time,
                        "working_time": sample.working_time,
                        "error": (
                            sample.error.model_dump(mode="json")
                            if sample.error
                            else None
                        ),
                    },
                }
            )

        sample_costs = [
            sample["execution"]["billing"]["total_cost"]
            for sample in samples
            if sample["execution"]["billing"]["total_cost"] is not None
        ]
        result = {
            "model": model_name,
            "condition": condition,
            "status": log.status,
            "created": log.eval.created,
            "task": {
                "name": log.eval.task,
                "arguments": log.eval.task_args,
                "plan": log.plan.model_dump(mode="json"),
                "metadata": log.eval.metadata,
            },
            "configuration": {
                "model": log.eval.model_generate_config.model_dump(mode="json"),
                "evaluation": log.eval.config.model_dump(mode="json"),
            },
            "inspect_log": (
                "../../../logs/"
                + Path(str(log.location).removeprefix("file://")).name
            ),
            "aggregate": (
                log.results.model_dump(mode="json") if log.results else None
            ),
            "usage": log.stats.model_dump(mode="json") if log.stats else None,
            "billing": {
                "source": "openrouter_response",
                "total_cost": sum(sample_costs) if sample_costs else None,
                "calls": sum(
                    sample["execution"]["billing"]["calls"] for sample in samples
                ),
            },
            "error": log.error.model_dump(mode="json") if log.error else None,
            "samples": samples,
        }
        (model_dir / "results.json").write_text(
            to_json(result, indent=2, inf_nan_mode="null").decode() + "\n",
            encoding="utf-8",
        )

        for sample in samples:
            summary_rows.append(
                {
                    "id": sample["id"],
                    "epoch": sample["epoch"],
                    "uuid": sample["uuid"],
                    "model": model_name,
                    "condition": condition,
                    "score": sample["judgement"]["score"],
                    "scored": sample["judgement"]["scored"],
                    "cost": sample["execution"]["billing"],
                    "timing": {
                        key: sample["execution"][key]
                        for key in (
                            "started_at",
                            "completed_at",
                            "total_time",
                            "working_time",
                        )
                    },
                    "error": sample["execution"]["error"],
                }
            )

    paired = []
    keys = sorted(
        {
            (row["model"], str(row["id"]), row["epoch"])
            for row in summary_rows
        }
    )
    for model, sample_id, epoch in keys:
        conditions = {
            row["condition"]: row
            for row in summary_rows
            if row["model"] == model
            and str(row["id"]) == sample_id
            and row["epoch"] == epoch
        }
        if {"raw", "agent"} <= conditions.keys():
            raw_score = conditions["raw"]["score"]
            agent_score = conditions["agent"]["score"]
            paired.append(
                {
                    "model": model,
                    "id": sample_id,
                    "epoch": epoch,
                    "raw": raw_score,
                    "agent": agent_score,
                    "difference": (
                        agent_score - raw_score
                        if isinstance(raw_score, (int, float))
                        and isinstance(agent_score, (int, float))
                        and math.isfinite(raw_score)
                        and math.isfinite(agent_score)
                        else None
                    ),
                }
            )

    differences = [
        row["difference"] for row in paired if row["difference"] is not None
    ]
    summary = {
        "samples": summary_rows,
        "comparison": {
            "paired": paired,
            "mean_difference": (
                sum(differences) / len(differences) if differences else None
            ),
            "agent_wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "raw_wins": sum(value < 0 for value in differences),
        },
        "cost": {
            "total": sum(row["cost"]["total_cost"] for row in summary_rows),
            "model": sum(row["cost"]["model_cost"] for row in summary_rows),
            "web_search": sum(
                row["cost"]["web_search_cost"] for row in summary_rows
            ),
        },
    }
    (run_dir / "summary.json").write_bytes(
        to_json(summary, indent=2, inf_nan_mode="null") + b"\n"
    )
    return summary
