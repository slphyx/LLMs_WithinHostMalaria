"""Run a paired raw-versus-agent research pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai import eval

from pricing import openrouter_costs
from research import TASK_IDS, agentic, raw
from research_results import export


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument(
        "--condition", choices=("raw", "agent", "both"), default="both"
    )
    parser.add_argument("--name", default="research-pilot")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set.")

    shared_config = json.loads(
        (ROOT / "config/benchmark.json").read_text(encoding="utf-8")
    )
    research_config = json.loads(
        (ROOT / "config/research.json").read_text(encoding="utf-8")
    )
    config = {
        **research_config,
        "judges": shared_config["judges"],
        "search_model": shared_config["agent"]["search_model"],
        "reasoning_effort": shared_config["agent"]["reasoning_effort"],
        "max_searches": shared_config["agent"]["max_searches"],
    }
    model = args.model or config["model"]
    sample_ids = args.sample_id or list(TASK_IDS)
    unknown = set(sample_ids) - set(TASK_IDS)
    if unknown:
        raise SystemExit(f"Unknown sample IDs: {', '.join(sorted(unknown))}")

    conditions = (
        ("raw", "agent") if args.condition == "both" else (args.condition,)
    )
    planned_limit = len(sample_ids) * (
        (config["raw_cost_limit"] if "raw" in conditions else 0)
        + (config["agent_cost_limit"] if "agent" in conditions else 0)
        + (
            config["max_searches"] * config["web_search_allowance"]
            if "agent" in conditions
            else 0
        )
    )
    if planned_limit > config["run_budget"]:
        raise SystemExit(
            f"Planned cost ceiling ${planned_limit:.2f} exceeds "
            f"${config['run_budget']:.2f}."
        )

    model_costs = openrouter_costs([model, *config["judges"]])
    tasks = []
    if "raw" in conditions:
        tasks.append(
            raw(
                judges=config["judges"],
                reasoning_effort=config["reasoning_effort"],
                judge_reasoning_effort=config["judge_reasoning_effort"],
                cost_limit=config["raw_cost_limit"],
            )
        )
    if "agent" in conditions:
        tasks.append(
            agentic(
                judges=config["judges"],
                search_model=config["search_model"],
                reasoning_effort=config["reasoning_effort"],
                judge_reasoning_effort=config["judge_reasoning_effort"],
                max_searches=config["max_searches"],
                cost_limit=config["agent_cost_limit"],
                token_limit=config["agent_token_limit"],
            )
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = re.sub(r"[^a-zA-Z0-9_-]+", "-", args.name).strip("-")
    run_dir = ROOT / "run_history" / f"{timestamp}_{label}"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    run_record = {
        "created": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "conditions": conditions,
        "sample_ids": sample_ids,
        "planned_cost_ceiling": planned_limit,
        "budget": config["run_budget"],
        "config": config,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_record, indent=2) + "\n", encoding="utf-8"
    )

    logs = eval(
        tasks,
        model=model,
        # Inspect's editor has optional fields, so its tool schema is non-strict.
        model_args={"strict_tools": False},
        # Inspect needs catalogue prices to enforce limits; logs retain charges.
        model_cost_config=model_costs,
        sample_id=sample_ids,
        log_dir=str(logs_dir),
        log_format="eval",
        max_tasks=config["max_tasks"],
        max_samples=config["max_samples"],
        max_sandboxes=config["max_samples"],
        log_refusals=True,
        metadata={"benchmark_run": run_dir.name},
    )
    summary = export(logs, run_dir)
    scores_complete = (
        len(summary["samples"]) == len(sample_ids) * len(conditions)
        and all(
            isinstance(row["score"], (int, float))
            and math.isfinite(row["score"])
            for row in summary["samples"]
        )
    )
    logs_complete = all(log.status == "success" for log in logs)
    if not logs_complete:
        status = "error"
    elif not scores_complete:
        status = "unscored"
    else:
        status = "success"
    run_record.update(
        {
            "completed": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "scores_complete": scores_complete,
            "actual_cost": summary["cost"]["total"],
            "summary": "summary.json",
            "logs": [
                Path(str(log.location).removeprefix("file://")).name for log in logs
            ],
        }
    )
    (run_dir / "run.json").write_text(
        json.dumps(run_record, indent=2) + "\n", encoding="utf-8"
    )
    print(run_dir)

    if summary["cost"]["total"] > config["run_budget"]:
        raise SystemExit("Run exceeded its stated budget; results were retained.")
    if status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
