---
name: scientific-modelling
description: Fit and criticise statistical, ODE, or Bayesian within-host models with SciPy and CmdStanPy when the task calls for computation.
---

# Scientific modelling

1. Inspect units, observation times, missing values, censoring, and the required output schema.
2. Start with the simplest defensible baseline. Add biological structure only when diagnostics or predictive evidence support it.
3. Use `scipy.optimize` for likelihood fitting and `scipy.integrate.solve_ivp` for ODEs. Use CmdStanPy when posterior simulation materially answers the question.
4. Separate latent biology from the observation model. Constrain positive quantities and state transformations.
5. Time one fit before adding starts or uncertainty runs. Bound this work; use observed information or a small bootstrap before attempting a profile likelihood.
6. Check identifiability, residuals, sensitivity to initial values, and held-out predictions where possible.
7. Report uncertainty and distinguish fitted associations from identified mechanisms.
8. Make `analysis.py` deterministic, relative-path based, responsible for regenerating `results.json`, and able to finish within 600 seconds.
