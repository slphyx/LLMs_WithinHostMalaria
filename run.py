"""Run the configured benchmark and write readable result files."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai import eval

from benchmark import malaria, malaria_agent
from pricing import openrouter_costs
from results import export_results


ROOT = Path(__file__).parent


def _baseline_comparison(summary: dict, baseline: dict) -> dict:
    """Join an agent-only run to an already stored raw result."""
    raw = {
        (str(row["id"]), row.get("epoch", 1)): row["judgement"]["score"]
        for row in baseline["samples"]
    }
    agent = {
        (str(row["id"]), row.get("epoch", 1)): row["score"]
        for row in summary["samples"]
    }
    paired = []
    for sample_id, epoch in sorted(raw.keys() & agent.keys()):
        raw_score = raw[(sample_id, epoch)]
        agent_score = agent[(sample_id, epoch)]
        difference = (
            agent_score - raw_score
            if isinstance(raw_score, (int, float))
            and isinstance(agent_score, (int, float))
            and math.isfinite(raw_score)
            and math.isfinite(agent_score)
            else None
        )
        paired.append(
            {
                "id": sample_id,
                "epoch": epoch,
                "raw": raw_score,
                "agent": agent_score,
                "difference": difference,
            }
        )
    differences = [
        row["difference"] for row in paired if row["difference"] is not None
    ]
    return {
        "paired": paired,
        "mean_raw": (
            sum(row["raw"] for row in paired) / len(paired)
            if paired
            else None
        ),
        "mean_agent": (
            sum(row["agent"] for row in paired) / len(paired)
            if paired
            else None
        ),
        "mean_difference": (
            sum(differences) / len(differences) if differences else None
        ),
        "agent_wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "raw_wins": sum(value < 0 for value in differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark.json")
    parser.add_argument("--model", action="append")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--raw-max-tokens",
        type=int,
        default=768,
        help="One-response token allowance for the raw condition.",
    )
    parser.add_argument(
        "--condition",
        choices=("raw", "agent", "both"),
        default="raw",
        help="Run direct answers, the agent setup, or a paired comparison.",
    )
    parser.add_argument(
        "--baseline-results",
        help=(
            "Stored raw results to reuse. Requires --condition agent and "
            "derives the model and sample IDs without rerunning raw."
        ),
    )
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    conditions = (
        ("raw", "agent")
        if args.condition == "both"
        else (args.condition,)
    )
    baseline_path = ROOT / args.baseline_results if args.baseline_results else None
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path
        else None
    )
    if baseline:
        if conditions != ("agent",):
            raise SystemExit(
                "--baseline-results requires --condition agent; raw will not run."
            )
        baseline_model = baseline["model"]
        respondents = args.model or [baseline_model]
        if respondents != [baseline_model]:
            raise SystemExit(
                "The requested model does not match the stored raw baseline."
            )
        baseline_ids = [str(row["id"]) for row in baseline["samples"]]
        sample_ids = args.sample_id or baseline_ids
        unknown = set(sample_ids) - set(baseline_ids)
        if unknown:
            raise SystemExit(
                "Sample IDs absent from the stored baseline: "
                + ", ".join(sorted(unknown))
            )
    else:
        respondents = args.model or config["respondents"]
        sample_ids = args.sample_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = re.sub(r"[^a-zA-Z0-9_-]+", "-", args.name).strip("-")
    run_dir = ROOT / "run_history" / "_".join(
        part for part in (timestamp, label) if part
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)

    run_record = {
        "created": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "selection": {
            "respondents": respondents,
            "conditions": conditions,
            "sample_id": sample_ids,
            "limit": args.limit,
            "epochs": args.epochs,
            "raw_max_tokens": args.raw_max_tokens,
        },
        "baseline_results": (
            str(baseline_path.relative_to(ROOT)) if baseline_path else None
        ),
    }
    run_file = run_dir / "run.json"
    run_file.write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tasks = []
    if "raw" in conditions:
        tasks.append(
            malaria(
                judges=config["judges"],
                max_tokens=args.raw_max_tokens,
            )
        )
    if "agent" in conditions:
        tasks.append(
            malaria_agent(
                judges=config["judges"],
                search_model=config["agent"]["search_model"],
                reasoning_effort=config["agent"][
                    "short_answer_reasoning_effort"
                ],
                max_searches=config["agent"]["max_searches"],
                cost_limit=config["agent"]["cost_limit"],
            )
        )

    eval_args = {}
    if "agent" in conditions:
        # Inspect's editor has optional fields, so its tool schema is non-strict.
        eval_args["model_args"] = {"strict_tools": False}
        eval_args["model_cost_config"] = openrouter_costs(
            [*respondents, *config["judges"]]
        )

    logs = eval(
        tasks,
        model=respondents,
        log_dir=str(logs_dir),
        log_format="eval",
        sample_id=sample_ids,
        limit=args.limit,
        epochs=args.epochs,
        max_tasks=config["max_tasks"],
        max_samples=config["max_samples"],
        max_sandboxes=config["max_sandboxes"],
        max_connections=config["max_connections"],
        log_refusals=True,
        metadata={"benchmark_run": run_dir.name},
        **eval_args,
    )
    summary = export_results(logs, run_dir)
    if baseline:
        summary["comparison"] = {
            "baseline_results": str(baseline_path.relative_to(ROOT)),
            **_baseline_comparison(summary, baseline),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    run_record["completed"] = datetime.now(timezone.utc).isoformat()
    logs_complete = all(log.status == "success" for log in logs)
    scores_complete = bool(summary["samples"]) and all(
        row["scored"] for row in summary["samples"]
    )
    run_record["status"] = (
        "error"
        if not logs_complete
        else "success"
        if scores_complete
        else "unscored"
    )
    run_record["scores_complete"] = scores_complete
    run_record["actual_cost"] = summary["cost"]["total"]
    run_record["summary"] = "summary.json"
    run_record["logs"] = [
        {
            "model": str(log.eval.model),
            "condition": log.eval.metadata.get("condition", "raw"),
            "status": log.status,
            "results": (
                "models/"
                + str(log.eval.model).replace("/", "__")
                + "/"
                + log.eval.metadata.get("condition", "raw")
                + "/results.json"
            ),
            "inspect_log": (
                "logs/"
                + Path(str(log.location).removeprefix("file://")).name
            ),
        }
        for log in logs
    ]
    run_file.write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(run_dir)

    if run_record["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
