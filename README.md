# Evaluating large language models for within-host malaria modelling

This repository contains the benchmark questions, model configurations, assessment procedures, code, and results supporting the manuscript **“Evaluating large language models for within-host malaria modelling.”**

The study investigates whether large language models (LLMs) can support specialised *Plasmodium falciparum* within-host modelling, including biological reasoning, executable modelling, and the development of testable model extensions. It compares conventional single responses with tool-enabled agentic workflows. A separate, exploratory experiment investigates whether an agentic LLM can propose a useful extension to an established gametocyte-dynamics model.

## Study overview

| Component | Design |
| --- | --- |
| LLM respondents | 8 models |
| Configurations | Single-response (`raw`) and agentic workflow (`agent`) |
| Short-answer benchmark | 55 questions across four categories |
| Extended benchmark | 3 synthetic-data computational modelling tasks |
| Short-answer evaluation | 3 independent LLM graders using question-specific criteria |
| Extended-task evaluation | 100-point hybrid assessment: deterministic checks and rubric-based scientific assessment |
| Exploratory open-research experiment | 15 agentic attempts; one Cao-derived model extension independently evaluated |

The **fixed benchmark** and the **exploratory open-research experiment** are distinct parts of the study. The benchmark compares performance on predefined tasks; the exploratory experiment examines an agent-generated modelling hypothesis without a predefined correct solution.

### Models and configurations

The eight respondent models were:

- GPT-5.6 Sol
- Claude Opus 5
- Gemini 3.6 Flash
- DeepSeek V4 Pro
- GPT-5.4 Mini
- DeepSeek R1
- Qwen 3.5 9B
- Ministral 3B

Each model was tested under the same two configurations:

- **Single-response (`raw`):** the model returns a direct answer without the external tools available to the agentic workflow.
- **Agentic workflow (`agent`):** the model can use literature/web search, a sandboxed computational environment, file operations, relevant guidance skills, and subagent delegation to develop and revise its answer.

The detailed agent setup is documented in [`agent/README.md`](agent/README.md), with the respondent and evaluator model identifiers in [`config/benchmark.json`](config/benchmark.json). See also [`MODEL_SELECTION.md`](MODEL_SELECTION.md) for the model-panel rationale.

## Benchmark tasks

### Short-answer questions (55)

The benchmark assesses four categories of within-host malaria modelling knowledge and reasoning:

1. Basic understanding
2. Applied reasoning
3. Synthesis and critique
4. Numerical calculation

Questions cover malaria biology, mathematical modelling, and interpretation of within-host parasite dynamics. The repository's question registry contains **58 entries in total**: the **55 short-answer questions** and **three separate extended computational tasks**. The source material comprises 38 supervisor-supplied and 20 self-authored entries; the registry total should not be confused with the number of short-answer questions used in the benchmark.

The questions, reference answers, and marking criteria are available in [`questions/`](questions/) and as a human-readable catalogue in [`build/full_short_answer_benchmark.pdf`](build/full_short_answer_benchmark.pdf). The reference answers represent acceptable scientific solutions rather than wording that respondents must reproduce exactly.

### Extended computational modelling tasks (3)

Three independent tasks use different synthetic datasets:

| Task | Modelling topic | Location |
| --- | --- | --- |
| SELF-C06 | Asexual parasite multiplication | [`tasks/SELF-C06`](tasks/SELF-C06) |
| SELF-C07 | Delayed gametocyte maturation | [`tasks/SELF-C07`](tasks/SELF-C07) |
| SELF-C08 | Human-to-mosquito infection | [`tasks/SELF-C08`](tasks/SELF-C08) |

Each task requires executable modelling work, structured numerical outputs, and biological interpretation. Task folders contain public inputs and output specifications; hidden evaluation targets are kept out of the respondent's task workspace. The same task specifications and assessment rules are applied to both configurations.

## How performance was assessed

### Short-answer grading

Short answers were evaluated by three LLM graders from different model families:

- GPT-5.5
- Claude Haiku 4.5
- Gemini 3.5 Flash

**The graders are separate from the respondents and do not use the agentic workflow being evaluated.** The same grading process is applied to single-response and agentic-workflow answers.

Each grader receives the question, a reference answer, predefined question-specific criteria, and the candidate response. Criteria are marked individually; scientifically equivalent answers may receive credit even if their wording or approach differs from the reference. The scoring code averages criterion-level coverage across valid grader judgements and applies a cap based on the most severe factual error identified. Mathematical or biological inaccuracies, unsupported mechanisms, and contradictions can reduce the score.

The factual-error caps are 1.00 for no substantive error, 0.50 for a minor error, 0.25 for a major error with some correct core material, and 0.00 when the central answer is wrong or contradicted. A response is left unscored if fewer than two valid grader judgements are returned. Full credit requires complete criterion coverage and no detected substantive error.

The scoring logic and grading criteria are available in [`benchmark.py`](benchmark.py), [`judging.py`](judging.py), and [`questions/rubrics.json`](questions/rubrics.json).

**Limitation:** LLM graders may be incorrect, especially when assessing open-ended scientific answers. Automated grading was not independently validated against human assessments in this study, so the scores should be interpreted with this limitation in mind.

### Extended-task grading

Each extended task is scored on a **100-point scale** using two complementary components:

- **50 points — deterministic checks:** executable outputs, hidden predictions, numerical consistency, uncertainty validity, and reproducibility checks, including independent code replay.
- **50 points — rubric-based assessment:** statistical choices, biological interpretation, research quality, and criticism of the proposed model, using separately invoked LLM graders.

This separates checks that can be reproduced computationally from scientific judgements that require interpretation. The extended-task scorer is implemented in [`research_scorer.py`](research_scorer.py), with additional documentation in [`agent/README.md`](agent/README.md).

## Main benchmark findings

The manuscript reports the following results for the fixed benchmark:

| Outcome | Single-response | Agentic workflow |
| --- | ---: | ---: |
| Mean short-answer score | 52.4% | 73.9% |
| Extended-task submissions meeting **all three** execution, completeness, and reproduction criteria | 0/24 | 12/24 |

Across the 440 model–question pairs, the agentic workflow scored higher in 188, tied in 198, and scored lower in 54. Sixteen of the 24 agentic computational submissions executed successfully, and 12 satisfied all three execution, completeness, and reproducibility criteria. GPT-5.6 Sol achieved the highest reported mean agentic short-answer score (95.1%) and extended-task score (94.1/100).

The comparison evaluates the **combined agentic configuration**; it does not isolate the effects of individual tools, subagents, or workflow components. Computational performance also does not by itself establish biological validity.

The completed benchmark records are under [`results/benchmark/`](results/benchmark/), including:

- [`run.json`](results/benchmark/run.json): evaluated model panel, task counts, and campaign metadata;
- [`summary.json`](results/benchmark/summary.json): aggregate results by model and configuration;
- [`scores.csv`](results/benchmark/scores.csv): selected per-task scores;
- [`costs.csv`](results/benchmark/costs.csv): recorded usage costs.

The result set covers **880 short-answer cells** (8 × 2 × 55) and **48 extended-task cells** (8 × 2 × 3), for **928 selected cells** overall. Additional saved attempts or replacements are recorded separately from these selected results.

## Exploratory open-research experiment

After the fixed benchmark, 15 agentic attempts were made to formulate, implement, and evaluate a model of the partially observed *P. falciparum* sexual-stage pathway:

- Five attempts targeted Fable 5.
- Ten attempts used GPT-5.6 Sol.

All 15 attempts were interrupted by automated safeguards. The five Fable attempts were rerouted immediately; the ten GPT-5.6 Sol attempts began but were subsequently interrupted. No attempt completed the planned autonomous research workflow.

Before its interruption, one GPT-5.6 Sol run proposed extending the published **Cao et al. gametocyte model** by allowing a participant-specific, partially pooled **effective male-allocation parameter**. The proposed model was treated as a hypothesis and was independently implemented and evaluated by researchers; it was **not** accepted as an autonomously validated biological discovery.

### Independent evaluation of the Cao-derived extension

The evaluation used publicly available data from **17 volunteers** in controlled human malaria infection studies. The models were fitted to total-parasite 18S measurements, female-marker Pfs25 and male-marker PfMGET measurements, with participant-specific piperaquine pharmacokinetic information.

The hierarchical extension was compared with a matched Cao-derived model using a common effective allocation parameter across participants. Predictive performance was assessed using participant-block Pareto-smoothed importance sampling (PSIS), withholding each participant's Pfs25/PfMGET measurement block while retaining their 18S and pharmacokinetic data.

| Predictive metric | Global-allocation comparator | Hierarchical extension |
| --- | ---: | ---: |
| Participant- and sex-equal mean negative log predictive density (NLPD; lower is better) | 4.656 | 4.636 |

The improvement was **0.020 per observation** (95% participant-bootstrap interval **0.012–0.029**), with the hierarchical model favoured in **15 of 17 participants**. This was a **small improvement in internal conditional sex-marker prediction**, not external or prospective validation. The effective allocation parameter should not be interpreted as a directly measured gametocyte sex ratio: sex-specific transcript abundance and assay characteristics can contribute to its value.

Related materials are in [`sex-cao-model/`](sex-cao-model/). The original model is described by Cao et al., *eLife* (2019), [“Modeling the dynamics of Plasmodium falciparum gametocytes in humans during malaria infection”](https://doi.org/10.7554/eLife.49058).

## Installation and running the benchmark

The scripts use Python and [uv](https://docs.astral.sh/uv/getting-started/installation/). The agentic configuration additionally uses Docker. Running the benchmark may incur API and search charges, and current provider availability or outputs may differ from the original evaluation.

Create the locked Python environment from the repository root:

```sh
uv sync --locked
```

Save your OpenRouter API key in a local `.env` file (which should not be committed):

```text
OPENROUTER_API_KEY=your_key_here
```

Run one short-answer question with one model:

```sh
uv run --env-file .env python run.py \
  --sample-id SELF-B03 \
  --model openrouter/qwen/qwen3.5-9b \
  --name test_question
```

Run the same question across the configured respondents:

```sh
uv run --env-file .env python run.py --sample-id SELF-B03 --name one-question
```

Run the complete single-response short-answer benchmark:

```sh
uv run --env-file .env python run.py --condition raw --name full
```

Build the agent container and run a paired comparison for one model:

```sh
docker compose -f agent/compose.yaml build
uv run --env-file .env python run.py \
  --condition both \
  --model openrouter/openai/gpt-5.6-sol \
  --name paired
```

Run the three extended computational tasks:

```sh
uv run --env-file .env python run_research.py --name paired-test
```

See [`agent/README.md`](agent/README.md) for additional options, task-level runs, and details of the computational environment. For an exploratory judge-stability check on selected items, the short-answer runner supports `--epochs 3`. This is not a substitute for expert validation.

**Re-running the software is not guaranteed to reproduce identical LLM outputs**, even with the original configurations, because hosted model versions, provider behaviour, and tool availability may change. The saved results document the original evaluation.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`config/benchmark.json`](config/benchmark.json) | Respondent and judge model panel; agent settings |
| [`questions/`](questions/) | Questions, reference answers, and marking criteria |
| [`tasks/`](tasks/) | Synthetic datasets, task contracts, and evaluation files |
| [`benchmark.py`](benchmark.py), [`judging.py`](judging.py) | Short-answer scoring and LLM judge-panel logic |
| [`research_scorer.py`](research_scorer.py) | Hybrid assessment of extended computational tasks |
| [`run.py`](run.py), [`run_research.py`](run_research.py) | Benchmark runners |
| [`agent/`](agent/) | Agentic workflow setup, tools, and sandbox configuration |
| [`results/benchmark/`](results/benchmark/) | Completed benchmark results and artefacts |
| [`build/full_short_answer_benchmark.pdf`](build/full_short_answer_benchmark.pdf) | Human-readable question, reference-answer, and rubric catalogue |
| [`sex-cao-model/`](sex-cao-model/) | Cao-derived model extension and related materials |
| [`MODEL_SELECTION.md`](MODEL_SELECTION.md) | Model-selection rationale |

## Interpretation and limitations

The findings support LLMs as **assistants to human-led malaria modelling**, rather than as autonomous substitutes for scientific expertise. In particular:

- LLM-based grading can contain errors and was not independently validated against human markings.
- The benchmark covers a defined set of tasks and model configurations; results should not be generalised uncritically to other settings.
- All open-research attempts were interrupted, preventing completion of the planned autonomous research processes.
- The Cao-derived extension was evaluated on the same cohort under conditional internal cross-validation; independent data and biological investigation are needed to assess generalisability and mechanistic meaning.

## Manuscript and data availability

**Manuscript:** Chung A, Aguas R, Gao B, Saralamba S. *Evaluating large language models for within-host malaria modelling.* Manuscript in preparation.

The benchmark datasets, implementations, assessment code, and saved results are provided in this repository. The underlying Cao study and its data should be cited separately when used.
