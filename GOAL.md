# Prompt: Discover a Step-Change Malaria Model

## Current task statement


Resolve this research problem completely:

> Discover, formulate, implement, and rigorously validate a genuinely step-change malaria model that advances beyond the strongest relevant existing models.

The primary goal is to materially improve modeling of the within-host *Plasmodium falciparum* sexual stage: the delayed, partially observed pathway from asexual parasites through sexual commitment and sequestered-but-developing immature gametocyte stages I-IV to circulating, non-replicating mature stage-V female and male gametocytes. This is a hidden maturation/sequestration problem, not hypnozoite dormancy, and the main inferential hinge is separating commitment, maturation delay and attrition, mature clearance, sex ratio, and assay effects from sparse or censored peripheral measurements.

Any scientifically defensible model class, data source, within-host scale, representation, or combination is eligible, including established and unanticipated methods. Mathematical novelty is not required if the advance is real and substantial. A failed implementation does not close its broader scientific approach.

The advance must materially improve prediction, explanation, inference, or decision. Do not privilege any model class, mechanism, timescale, or data type within that scope in advance. A complete result must decisively beat the strongest relevant comparators on evidence not used to design or select the candidate, survive appropriate adversarial audit, and be independently reproducible. Make only the claims the evidence supports.

Assume for purposes of this task that a complete affirmative solution satisfying these requirements exists.

Partial progress does not count unless it implies exactly the complete resolution above. Development fit, synthetic-only success, a weak-comparator or post hoc win, a non-executable story, a tiny gain, an unresolved necessary hinge, or an unconfirmed candidate is insufficient. Treat a reparameterization, ensemble, prior, transformation, or optimization change as cosmetic only when analysis shows that it is.


## Search mandate

Use subagents aggressively and dynamically. You have up to 64 concurrent agents available. Do not use a fixed assignment such as "N agents for strategy X." Instead, manage the search using the following heuristics:

- Begin with a genuinely diverse portfolio of approaches spanning different formulations, assumptions, information sources, scales, model classes, and validation strategies. Examples are never exhaustive.
- Preserve independence during early rounds. Give most agents the task and protected-data boundaries without the favored mechanism, detailed local failure history, or other agents' proposals. Require independent route cards before exposing `literature/approach_family_evidence.md`.
- Maintain a compact registry of approach families grouped by substantive scientific or mathematical idea, not wording, software, or optimizer. Record only each family's concrete formulation, expected advantage, falsifier, blocking hinge, and status. Redirect duplicates toward underexplored families; the registry is a search-diversity tool, not an administrative system.
- Do not let elegance, fashion, intuition, or early development success make one approach dominate. A route ending at a theorem, mechanism, measurement, identification condition, or validation bridge necessary to its claim is blocked unless it supplies a genuinely new resolution.
- Keep several incompatible routes alive through multiple rounds. Cross-pollinate only after independent routes are developed enough to expose their actual strengths and gaps.
- Require concrete equations or algorithms, executable constructions, predictions, falsifiers, counterexamples, ablations, or tests. Reject status reports, vague optimism, architecture names, and claims that a difficult step is "routine."
- Use adversarial agents throughout to test leakage, equivalence, comparator fairness, post hoc choices, confounding, numerical failure, implausible behavior, dependence on a single dataset, unsupported interpretation, and model-appropriate edge cases. Match audits to the claim; biological identification is not required for a purely predictive result.
- The root agent should repeatedly synthesize, challenge, redirect, and launch new rounds. Do not stop after the first wave fails. Reopen blocked routes only when something materially new addresses the exact gap.

## Frontier research

Search the public research frontier aggressively and repeatedly. Begin broadly, then search each route's exact questions. Examine papers, preprints, dissertations, trials, datasets, supplementary material, research code, negative results, and adjacent fields. Build a dated frontier map of the strongest models, equations or algorithms, evidence, assumptions, limitations, contradictions, and open hinges. Use primary sources and verify versions and reused data.

Use the frontier to avoid rediscovering failed ideas and to combine promising components into new solutions. Extract concrete methods and failure modes, not citation lists. Search for new development and independent evaluation data. Public literature never substitutes for implementation and validation and never authorizes protected-data use.

## Evidence standard

Use development evidence freely. Evaluate surviving candidates against the strongest relevant models on held-out, external, temporal, geographic, trial-level, prospective, or experimental evidence that was not used to design or select them.

Give comparators fair information and tuning. New modalities are allowed: distinguish model, data, and interaction gains where meaningful without disqualifying a real model-plus-data breakthrough. Do not repeatedly tune a candidate to the same held-out evidence.

## Completion rule

Do not return merely because current approaches fail, agents report theorem-strength gaps, eligible data are difficult to obtain, or clean local scoring opportunities are exhausted. Continue launching new rounds, pursuing frontier leads, seeking eligible evidence, and developing fresh formulations.

Return only when a complete affirmative step-change model has been found, implemented, validated on independent held-out evidence, independently reproduced, and has survived adversarial audit. Return the model, exact supported claim, equations or algorithm, evidence, comparator results, audits, and reproduction instructions.

Do not return a reduction, partial result, isolated missing hinge, bounded-negative or inconclusive substitute, "best effort" summary, unvalidated candidate, or explanation of why the problem is difficult.

Spend at least 12 hours on this before even thinking of returning or giving up.
