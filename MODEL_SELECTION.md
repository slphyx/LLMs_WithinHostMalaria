# Model selection


## Respondents

The configured set is a capability and size spread:

- Frontier proprietary — GPT-5.6 Sol, Claude Opus 5
- Efficient proprietary — Gemini 3.6 Flash, GPT-5.4 Mini
- Large open-weight — DeepSeek V4 Pro, DeepSeek R1 0528
- Small open-weight — Qwen3.5 9B, Ministral 3 3B

Current model IDs and availability:

- https://openrouter.ai/openai/gpt-5.6-sol
- https://openrouter.ai/anthropic/claude-opus-5
- https://openrouter.ai/google/gemini-3.6-flash
- https://openrouter.ai/deepseek/deepseek-v4-pro
- https://openrouter.ai/openai/gpt-5.4-mini
- https://openrouter.ai/deepseek/deepseek-r1-0528
- https://openrouter.ai/qwen/qwen3.5-9b
- https://openrouter.ai/mistralai/ministral-3b-2512

## Judges

GPT-5.5, Claude Haiku 4.5, and Gemini 3.5 Flash form a mixed-family panel. They
are kept out of the respondent list so every answer receives three separate
marks. GPT-5.5 and Claude use low reasoning; a controlled check found that low
reasoning changed Gemini's verdicts more substantially, so Gemini retains its
default. The reference answer and atomic criteria constrain the task;
temperature zero reduces avoidable variation.

Claude Sonnet 5 was removed after it refused 27 of 50 benign within-host malaria
marking requests in a pilot, despite direct Anthropic routing. Those refusals
were recorded as missing judgements rather than incorrect answers. On the same
ten difficult saved answers, Sonnet 4.6 refused three and Opus 4.8 refused five;
Haiku 4.5 returned ten valid verdicts. Haiku is therefore the provisional
Claude-family judge, but its accuracy and agreement with expert markers still
require human validation.

Current judge model :

- https://openrouter.ai/openai/gpt-5.5
- https://openrouter.ai/anthropic/claude-haiku-4.5
- https://openrouter.ai/google/gemini-3.5-flash
