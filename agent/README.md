# Agent setup

This directory contains the reusable capability setup, not a benchmark or a
grader. [`setup.py`](setup.py) constructs Inspect's Deep Agent with Git, Bash,
a text editor, Python, scientific packages, web search, and research skills.
The benchmark suite supplies the samples, output contract, and scorer.

The main benchmark compares the same 55 short-answer samples in two conditions:

- `raw`: direct model generation;
- `agent`: the setup in this directory.

Both use the same scorer from `benchmark.py`. The scorer does not know which
setup produced an answer, apart from condition metadata retained for reporting.

## Run

From the repository root:

```sh
uv sync --locked
docker compose -f agent/compose.yaml build
uv run --env-file .env python run.py \
  --condition both \
  --model openrouter/openai/gpt-5.6-terra \
  --name paired
```

Put only this in `.env`:

```text
OPENROUTER_API_KEY=...
```



Use `--sample-id` for selected questions, `--condition raw` or
`--condition agent` for one condition, and `--model` to override respondents.
Shared models, judges, parallelism, and agent limits live in
`config/benchmark.json`.

## Extended computational modelling tasks

The three extended tasks have a different output contract: executable code,
reproduced numerical artefacts, and a report. They therefore have a separate
hybrid evaluator in `research_scorer.py`, owned by that task suite rather than
by the agent setup. Its raw and agent conditions still use that exact same
evaluator.

Run them with:

```sh
uv run --env-file .env python run_research.py --name paired-pilot
```

Use `--sample-id SELF-C07` for one research task. Its settings live in
`config/research.json`; shared judge and agent settings remain in
`config/benchmark.json`.



Each extended-task score combines:

- 50 deterministic points for executable, reproducible results and hidden
  predictions;
- 50 points from three separately invoked model judges applying atomic
  scientific criteria.

`analysis.py` is replayed in a fresh directory containing only the submitted
code and public inputs. Code owns all weights and caps; judges return only
criterion findings and short reasons. Alternative defensible
models can receive full credit.

## Results

The benchmark result contains both conditions in the following layout:

```text
results/benchmark/models/<model>/
├── raw/executable/<task>/
└── agent/executable/<task>/
```





- BLADE: https://doi.org/10.18653/v1/2024.findings-emnlp.815
- ScienceAgentBench: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f12b4df26344f3be803c06b555252efe-Abstract-Conference.html
- PaperBench: https://proceedings.mlr.press/v267/starace25a.html
- Inspect Deep Agent: https://inspect.aisi.org.uk/deepagent.html
- Inspect sandboxing: https://inspect.aisi.org.uk/sandboxing.html
- OpenRouter web search: https://openrouter.ai/docs/guides/features/server-tools/web-search
- OpenRouter usage accounting: https://openrouter.ai/docs/cookbook/administration/usage-accounting
