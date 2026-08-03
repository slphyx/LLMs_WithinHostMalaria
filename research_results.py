"""Readable projections of extended-task evaluation logs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from inspect_ai.log import EvalLog
from pydantic_core import to_json

from billing import model_billing


def export(logs: list[EvalLog], run_dir: Path) -> dict:
    rows = []
    for log in logs:
        condition = log.eval.metadata["condition"]
        for sample in log.samples or []:
            score = next(iter(sample.scores.values())) if sample.scores else None
            metadata = score.metadata if score and score.metadata else {}
            store = sample.store or {}
            model_cost, calls = model_billing(sample)
            web_cost = metadata.get("web_search_cost", 0)
            artifacts = metadata.get("artifacts", {})
            sample_dir = run_dir / "conditions" / condition / str(sample.id)
            sample_dir.mkdir(parents=True, exist_ok=True)
            response = sample.output.completion if sample.output else ""
            if response:
                (sample_dir / "response.txt").write_text(
                    response, encoding="utf-8"
                )

            for name, content in artifacts.items():
                output_name = name.removeprefix("reproduced_")
                is_submitted_numerical = (
                    name in ("results.json", "trajectory.csv")
                    and f"reproduced_{name}" in artifacts
                )
                if content and not is_submitted_numerical:
                    (sample_dir / output_name).write_text(content, encoding="utf-8")

            tool_counts = Counter(
                event.function
                for event in sample.events
                if getattr(event, "event", None) == "tool"
            )
            record = {
                "id": sample.id,
                "condition": condition,
                "model": str(log.eval.model),
                "score": score.value if score else None,
                "explanation": score.explanation if score else None,
                "response_file": "response.txt" if response else None,
                "artifact_error": (
                    metadata.get("artifact_error")
                    or store.get("artifact_error")
                ),
                "automated": metadata.get("automated"),
                "scientific": metadata.get("scientific"),
                "execution": metadata.get("execution"),
                "judges": metadata.get("judges"),
                "judge_failures": metadata.get("judge_failures"),
                "web_search_calls": metadata.get("web_search_calls", []),
                "usage": {
                    name: usage.model_dump(mode="json")
                    for name, usage in sample.model_usage.items()
                },
                "tools": dict(tool_counts),
                "cost": {
                    "model": model_cost,
                    "web_search": web_cost,
                    "total": model_cost + web_cost,
                    "model_calls": calls,
                },
                "timing": {
                    "started_at": sample.started_at,
                    "completed_at": sample.completed_at,
                    "total_time": sample.total_time,
                    "working_time": sample.working_time,
                },
                "error": (
                    sample.error.model_dump(mode="json") if sample.error else None
                ),
            }
            (sample_dir / "score.json").write_bytes(
                to_json(record, indent=2, inf_nan_mode="null") + b"\n"
            )
            rows.append(record)

    paired = []
    for sample_id in sorted({str(row["id"]) for row in rows}):
        conditions = {
            row["condition"]: row for row in rows if str(row["id"]) == sample_id
        }
        if {"raw", "agent"} <= conditions.keys():
            raw_score = conditions["raw"]["score"]
            agent_score = conditions["agent"]["score"]
            paired.append(
                {
                    "id": sample_id,
                    "raw": raw_score,
                    "agent": agent_score,
                    "difference": (
                        agent_score - raw_score
                        if isinstance(raw_score, (int, float))
                        and isinstance(agent_score, (int, float))
                        else None
                    ),
                }
            )

    differences = [
        row["difference"] for row in paired if row["difference"] is not None
    ]
    summary = {
        "samples": rows,
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
            "total": sum(row["cost"]["total"] for row in rows),
            "model": sum(row["cost"]["model"] for row in rows),
            "web_search": sum(row["cost"]["web_search"] for row in rows),
        },
    }
    (run_dir / "summary.json").write_bytes(
        to_json(summary, indent=2, inf_nan_mode="null") + b"\n"
    )
    return summary
