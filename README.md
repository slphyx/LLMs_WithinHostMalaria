# Within-host malaria modelling benchmark

A benchmark of postgraduate-level understanding and application of within-host
*Plasmodium falciparum* modelling.

- 38 question–answer pairs were supplied by the project supervisor.
- 20 are self-authored and include literature-backed answer metadata.
- Five self-authored questions require exact computational results.
- Three extended computational modelling tasks require code, structured results, and a report.
- Questions progress from basic understanding to applied, synthetic, and computational work.
- The supervisor's question and answer text is unchanged.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
create the locked environment:

```sh
uv sync --locked
```

Store the OpenRouter key in the ignored `.env` file:

```text
OPENROUTER_API_KEY=...
```

## Run

The respondent and judge panels are in
[`config/benchmark.json`](config/benchmark.json). Start with one easy question
and one respondent:

```sh
uv run python run.py \
  --sample-id SELF-B03 \
  --model openrouter/qwen/qwen3.5-9b \
  --name test_question
```

Run the same question across every configured respondent:

```sh
uv run python run.py --sample-id SELF-B03 --name one-question
```

Run the complete raw benchmark:

```sh
uv run python run.py --name full
```

Run a paired comparison of the same model in raw and agent mode:

```sh
docker compose -f agent/compose.yaml build
uv run --env-file .env python run.py \
  --condition both \
  --model openrouter/openai/gpt-5.6-terra \
  --name paired
```

Both modes receive the same 55 samples and are marked by the same
judge setup. Only the solver changes: `raw` generates a direct
answer, while `agent` supplies the reusable tools, skills, sandbox, and web
search setup in [`agent/setup.py`](agent/setup.py).

The three extended computational modelling tasks remain a separate setup because their
output and replay checks require case by case handling.



## Results

The completed evaluation will look something like this:

```text
results/benchmark/
├── run.json
├── summary.json
├── scores.csv
├── costs.csv
├── short/summary.json
├── executable/summary.json
└── models/<model>/<condition>/
    ├── short/results.json
    └── executable/<task>/
```

`summary.json` joins both conditions across all eight models and both benchmark
suites. `scores.csv` contains the 928 selected model–condition–task cells;
`costs.csv` accounts for all 959 persisted attempts, including replacements.
The model directories contain the selected readable short-answer records and
extended-task artefacts.

The complete human-readable question, reference-answer and rubric catalogue is
retained as [`build/full_short_answer_benchmark.pdf`](build/full_short_answer_benchmark.pdf).
It is a repository artefact, not an assessed supplementary-text submission.

### Open-research attempts

The subsequent open-research phase comprised 15 attempts: five targeting
Fable 5 and ten using GPT-5.6 Sol. Safeguards stopped all five Fable attempts
immediately and eventually interrupted all ten GPT-5.6 Sol attempts; the
Cao-derived candidate was the only substantive result produced before an
interruption. These are observed operational outcomes, not a controlled
estimate of safeguard performance.

For a judge-stability check, repeat the same selected items with
`--epochs 3`.




## Marking method

Each answer is assessed against the question-specific criteria in
[`questions/rubrics.json`](questions/rubrics.json). The three judges run in
parallel and receive:

1. the question;
2. the reference answer;
3. the atomic criteria; and
4. the anonymous candidate answer.

The reference is explicitly described as one valid formulation, not a wording
template. Scientifically equivalent answers are accepted, and omissions are
distinguished from false claims.

Each judge returns one boolean per positive criterion and one overall factual
error level. Reasons are limited to short audit notes. Python calculates
coverage from the criterion votes and applies the strictest valid error finding:

- no substantive error: 1.00;
- minor local error: 0.50;
- major error with some correct core material: 0.25; and
- central answer wrong or contradicted: 0.00.

Full credit requires complete coverage and no detected
substantive error. A response is unscored when fewer than two judges return
valid verdicts.

The approach follows marking in MT-Bench and Prometheus,
positive and negative criteria in HealthBench, criterion-level grading in
PaperBench, structured decisions in G-Eval, and diverse judge panels in PoLL.

Respondents are asked for a direct answer. Conceptual short answers are capped
at 768 output tokens; exact calculations and extended tasks use 12,000.


## Extended computational modelling tasks

[`tasks/SELF-C06`](tasks/SELF-C06), [`tasks/SELF-C07`](tasks/SELF-C07), and
[`tasks/SELF-C08`](tasks/SELF-C08) cover parasite growth, delayed gametocyte
ODE modelling, and human-to-mosquito transmission. Each contains public
synthetic data, an exact output contract, hidden targets, and a 100-point
rubric while permitting defensible alternative models.

Half of the score is deterministic: output validity, hidden prediction error,
numerical consistency, and uncertainty validity. The remaining criteria cover
statistical choices, biological interpretation, research quality, and model
criticism. Only `input/`, `README.md`, and `result.example.json` enter the
model workspace; hidden grading files remain outside it.

The extended-task suite is documented in
[`agent/README.md`](agent/README.md). Run all three tasks with:

```sh
docker compose -f agent/compose.yaml build
uv run --env-file .env python run_research.py --name paired-test
```

## Files

```text
benchmark.py                        shared short-answer samples, scorer, conditions
judging.py                          task-agnostic judge-panel plumbing
run.py                              raw/agent short-answer runner
results.py                          paired projection of short-answer logs
config/benchmark.json              respondent, judge, and parallelism settings
questions/combined_questions.json  canonical 58-question registry
questions/computational_questions.json exact computational source set
questions/model_fitting_questions.json three extended-task records
questions/rubrics.json             114 atomic marking criteria
questions/supervisor_questions.json supervisor-supplied source set
questions/self_questions.json      self-authored source set and references
tasks/SELF-C06...SELF-C08/          research inputs, contracts, and graders
research.py                         raw/agent extended-task conditions
research_scorer.py                  hybrid extended-task evaluator
run_research.py                     extended-task runner
agent/                              reusable solver, image, tools, and skills only
MODEL_SELECTION.md                  model-panel rationale
```
