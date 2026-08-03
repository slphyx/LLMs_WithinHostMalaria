"""Shared judge-panel plumbing for every benchmark task type."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from inspect_ai.model import GenerateConfig, ModelOutput, get_model


def judge_names(value: str | list[str]) -> list[str]:
    """Normalize and validate a diverse judge panel."""
    names = value if isinstance(value, list) else value.split(",")
    names = [name.strip() for name in names if name.strip()]
    if len(names) < 3 or len(names) != len(set(names)):
        raise ValueError("Supply at least three distinct judge models.")
    return names


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a single Markdown code fence."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object.")
    return value


def judge_config(
    *,
    name: str,
    system_message: str,
    reasoning_effort: str | None = None,
) -> GenerateConfig:
    """Build the common deterministic configuration for a judge."""
    extra_body = (
        {"provider": {"only": ["anthropic"]}}
        if name.startswith("openrouter/anthropic/")
        else None
    )
    return GenerateConfig(
        system_message=system_message,
        temperature=0,
        seed=1,
        reasoning_effort=reasoning_effort,
        max_tokens=8000,
        max_connections=12,
        extra_body=extra_body,
    )


def make_panel(
    names: list[str],
    *,
    system_message: str,
    reasoning_effort: str | Callable[[str], str | None] | None = None,
) -> list[tuple[str, Any]]:
    """Instantiate judges once when an Inspect scorer is constructed."""
    judges = []
    for name in names:
        effort = (
            reasoning_effort(name)
            if callable(reasoning_effort)
            else reasoning_effort
        )
        judges.append(
            (
                name,
                get_model(
                    name,
                    config=judge_config(
                        name=name,
                        system_message=system_message,
                        reasoning_effort=effort,
                    ),
                ),
            )
        )
    return judges


def _output_metadata(output: ModelOutput) -> dict[str, Any]:
    data = output.model_dump(mode="json")
    return {
        "served_model": data["model"],
        "usage": data["usage"],
        "generation_time": data["time"],
        "model_metadata": data["metadata"],
    }


async def run_panel(
    judges: list[tuple[str, Any]],
    *,
    candidate_model: str,
    data: dict[str, Any],
    parse: Callable[[str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run eligible judges and return valid findings plus auditable failures."""
    eligible = [
        (name, model) for name, model in judges if name != candidate_model
    ]
    outputs = await asyncio.gather(
        *(model.generate(json.dumps(data)) for _, model in eligible),
        return_exceptions=True,
    )

    verdicts, failures = [], []
    for (name, _), output in zip(eligible, outputs):
        try:
            if isinstance(output, BaseException):
                raise output
            verdicts.append(
                {
                    "judge": name,
                    **_output_metadata(output),
                    **parse(output.completion),
                }
            )
        except Exception as error:
            failure = {"judge": name, "error": str(error)}
            if not isinstance(output, BaseException):
                failure.update(
                    {
                        **_output_metadata(output),
                        "response": output.completion,
                        "stop_reason": output.stop_reason,
                    }
                )
            failures.append(failure)
    return verdicts, failures
