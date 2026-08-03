# SELF-C06: fit an untreated parasite trajectory

Fit the simplest scientifically defensible model to
[`input/parasitaemia.csv`](input/parasitaemia.csv). Predict parasite density at
the times in [`input/prediction_times.csv`](input/prediction_times.csv).

The data are synthetic pre-treatment qPCR measurements from one
*Plasmodium falciparum* infection. `ND` means that the measurement was below
the detection limit of 10 parasites/mL; it is left-censored, not zero.

Estimate the average parasite multiplication factor per 48-hour cycle with an
uncertainty interval. Use a free-intercept log-linear model as the baseline,
then decide whether a more complex model is justified by diagnostics and
predictive evidence.

Create:

- `analysis.py`, which reads only the supplied input files and reproduces the
  submitted results;
- `results.json`, following [`result.example.json`](result.example.json); and
- `report.md`, containing at most 300 words.

`analysis.py` must finish from a clean workspace within 600 seconds on two
CPU cores. Exceeding this limit is recorded as a clean-replay execution error
and caps the total score at 50.

Model choice is not prescribed. Marks reward reproducibility, held-out
prediction, statistical validity, biological interpretation, and justified
parsimony.

The evaluator runs outside the agent workspace. Files under `grading/` must not
be exposed to the agent.
