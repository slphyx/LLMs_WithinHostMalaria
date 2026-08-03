---
name: git-workflow
description: Use Git to inspect, checkpoint, and audit a modelling analysis without exposing hidden evaluator files.
---

# Git workflow

- Begin with `git status --short` and inspect the supplied files.
- Commit meaningful working states so failed experiments can be compared or reversed.
- Use concise commit messages that state the modelling decision.
- Never rewrite shared history or add generated caches, environments, or secrets.
- Finish with `git diff --stat`, `git status --short`, and one reproducibility run.
