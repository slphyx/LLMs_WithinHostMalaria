"""The reusable agent solver, independent of datasets and grading."""

from __future__ import annotations

from pathlib import Path

from inspect_ai.agent import deepagent
from inspect_ai.solver import Solver
from inspect_ai.tool import bash, python, text_editor

from .web import openrouter_web_search


ROOT = Path(__file__).resolve().parents[1]
SKILLS = sorted((ROOT / "agent/skills").iterdir())


def agent_solver(
    *,
    search_model: str,
    max_searches: int,
    instructions: str,
) -> Solver:
    """Create the same tool-using setup for any compatible benchmark suite."""
    return deepagent(
        tools=[
            bash(timeout=600),
            python(timeout=600),
            text_editor(timeout=600),
        ],
        web_search=openrouter_web_search(search_model, max_searches),
        skills=SKILLS,
        memory=False,
        # Writing subagents share one workspace, so dispatch stays synchronous.
        background=False,
        max_depth=1,
        instructions=instructions,
    )
