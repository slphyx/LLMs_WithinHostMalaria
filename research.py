"""Extended computational modelling suite with raw and agent conditions."""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message
from inspect_ai.util import JSONSchema, sandbox

from agent.setup import agent_solver
from judging import judge_names
from research_scorer import research_scorer


ROOT = Path(__file__).resolve().parent
TASK_IDS = ("SELF-C06", "SELF-C07", "SELF-C08")

RAW_PROMPT = """\
Complete the extended computational modelling task in one response without tools or follow-up. Return
exactly the requested structured fields: analysis_py as plain Python,
results as a JSON object, trajectory_csv as plain CSV, and report_md as plain
Markdown. Do not add code fences. Use an empty trajectory_csv string when it
is not requested. analysis.py must read the supplied input paths and
reproduce every numerical artefact when run.
"""

SUBMISSION_SCHEMA = ResponseSchema(
    name="research_submission",
    strict=True,
    json_schema=JSONSchema(
        type="object",
        properties={
            "analysis_py": JSONSchema(type="string"),
            "results": JSONSchema(type="object", additionalProperties=True),
            "trajectory_csv": JSONSchema(type="string"),
            "report_md": JSONSchema(type="string"),
        },
        required=[
            "analysis_py",
            "results",
            "trajectory_csv",
            "report_md",
        ],
        additionalProperties=False,
    ),
)


def research_samples(condition: str) -> list[Sample]:
    samples = []
    for task_id in TASK_IDS:
        task_dir = ROOT / "tasks" / task_id
        public_files = [
            task_dir / "README.md",
            task_dir / "result.example.json",
            *sorted((task_dir / "input").iterdir()),
        ]
        prompt = "\n\n".join(
            [
                (task_dir / "README.md").read_text(encoding="utf-8"),
                "## Supplied files",
                *[
                    f"### {path.relative_to(task_dir)}\n```\n"
                    f"{path.read_text(encoding='utf-8')}\n```"
                    for path in public_files[1:]
                ],
            ]
        )
        samples.append(
            Sample(
                id=task_id,
                input=prompt,
                metadata={"condition": condition},
                files={
                    "README.md": str(task_dir / "README.md"),
                    "result.example.json": str(task_dir / "result.example.json"),
                    "input": str(task_dir / "input"),
                },
                setup=(
                    "git init -q\n"
                    "git config user.name benchmark\n"
                    "git config user.email benchmark@local\n"
                    "git add README.md result.example.json input\n"
                    'git commit -qm "Initial task"\n'
                ),
            )
        )
    return samples


@solver
def raw_submission() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(
            state,
            tool_calls="none",
            response_schema=SUBMISSION_SCHEMA,
        )
        try:
            submission = json.loads(state.output.completion)
            await sandbox().write_file("analysis.py", submission["analysis_py"])
            await sandbox().write_file(
                "results.json",
                json.dumps(
                    submission["results"], ensure_ascii=False, indent=2
                )
                + "\n",
            )
            if submission["trajectory_csv"].strip():
                await sandbox().write_file(
                    "trajectory.csv", submission["trajectory_csv"]
                )
            await sandbox().write_file("report.md", submission["report_md"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            state.store.set("artifact_error", str(error))
        return state

    return solve


@task
def raw(
    judges: str | list[str],
    reasoning_effort: str = "low",
    judge_reasoning_effort: str = "low",
    cost_limit: float = 0.75,
) -> Task:
    names = judge_names(judges)
    return Task(
        dataset=research_samples("raw"),
        solver=[system_message(RAW_PROMPT), raw_submission()],
        scorer=research_scorer(names, judge_reasoning_effort),
        sandbox=("docker", str(ROOT / "agent/compose.yaml")),
        config=GenerateConfig(
            reasoning_effort=reasoning_effort,
            max_tokens=12000,
            max_connections=3,
        ),
        cost_limit=cost_limit,
        time_limit=1200,
        fail_on_error=False,
        metadata={
            "condition": "raw",
            "suite": "research",
            "judges": names,
        },
    )


@task
def agentic(
    judges: str | list[str],
    search_model: str,
    reasoning_effort: str = "low",
    judge_reasoning_effort: str = "low",
    max_searches: int = 6,
    cost_limit: float = 1.75,
    token_limit: int = 500000,
) -> Task:
    names = judge_names(judges)
    return Task(
        dataset=research_samples("agent"),
        solver=agent_solver(
            search_model=search_model,
            max_searches=max_searches,
            instructions=(
                "Complete the extended computational modelling task in the workspace. Produce analysis.py, "
                "results.json, and report.md exactly as requested. Run the analysis "
                "and inspect every artefact before submitting."
            ),
        ),
        scorer=research_scorer(names, judge_reasoning_effort),
        sandbox=("docker", str(ROOT / "agent/compose.yaml")),
        config=GenerateConfig(
            reasoning_effort=reasoning_effort,
            max_tokens=12000,
            max_connections=3,
        ),
        message_limit=50,
        token_limit=token_limit,
        cost_limit=cost_limit,
        time_limit=1800,
        fail_on_error=False,
        metadata={
            "condition": "agent",
            "suite": "research",
            "judges": names,
            "search_model": search_model,
        },
    )
