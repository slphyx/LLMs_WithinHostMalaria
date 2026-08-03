# SELF-C07: model delayed gametocyte appearance

The supplied files describe one synthetic controlled *Plasmodium falciparum*
infection:

- [`input/asexual_driver.csv`](input/asexual_driver.csv) is a pre-estimated
  circulating asexual-parasite trajectory;
- [`input/gametocytes.csv`](input/gametocytes.csv) contains mature
  female-gametocyte measurements analogous to Pfs25 qRT-PCR; and
- [`input/prediction_times.csv`](input/prediction_times.csv) gives the
  measurements withheld for evaluation.

`ND` is left-censored at 2 gametocytes/mL, not zero. Treatment starts on day
8. For this task, assume that it removes asexual parasites but has no direct
effect on immature or mature gametocytes.

Fit a non-negative ODE model linking asexual density to delayed appearance of
circulating mature female gametocytes. It must contain at least one
sequestered immature compartment and a mature compartment with maturation
inflow and first-order loss. The number of immature compartments and
production function are your choice. Treat the asexual trajectory as a fixed
input and state how it is interpolated.

Estimate the mean immature maturation time, effective production coefficient,
and mature-gametocyte clearance rate. Report the corresponding mature
half-life with an uncertainty interval. Compare your model with at least one
simpler or alternative maturation structure. Use an adaptive ODE solver and
report its tolerances.

Create:

- `analysis.py`, which reads only the supplied numerical inputs and reproduces
  the analysis;
- `results.json`, following [`result.example.json`](result.example.json);
- `trajectory.csv`, with `time_days` and
  `mature_gametocytes_per_ml` columns at intervals no wider than 0.1 day; and
- `report.md`, containing at most 400 words and citations for biological
  assumptions.

`analysis.py` must finish from a clean workspace within 600 seconds on two
CPU cores. Exceeding this limit is recorded as a clean-replay execution error
and caps the total score at 50.

Model choice is not otherwise prescribed. Marks reward executable analysis,
held-out prediction, statistical validity, biological interpretation, and
recognition of weakly identified parameters.

Useful starting points are Cao et al.
([doi:10.7554/eLife.49058](https://doi.org/10.7554/eLife.49058)) and Reuling
et al. ([doi:10.7554/eLife.31549](https://doi.org/10.7554/eLife.31549)).
External literature may inform assumptions or priors, but no external
participant-level observations may be added to this profile.

The evaluator runs outside the agent workspace. Files under `grading/` must
not be exposed to the agent.
