# Cao-derived hierarchical allocation extension

This directory is a compact, reproducible implementation of a matched
comparison derived from the Cao et al. within-host malaria model. The
global-allocation comparator uses one effective male allocation for every
participant. The hierarchical-allocation extension permits persistent,
partially pooled participant variation:

```text
logit(rho_i) = logit(rho) + tau_rho z_i
z_i ~ Normal(0, 1); tau_rho ~ half-Normal(0, 0.5)
female source = (1-rho_i) S_i(t); male source = rho_i S_i(t)
```

The source split conserves mass. The Cao recurrence, piperaquine PK/PD,
female/male maturation and clearance, total-18S likelihood, and Pfs25/PfMGET
observation model are otherwise identical between the comparator and extension.

## Qualified result

The primary endpoint jointly deletes **all** Pfs25 and PfMGET measurements for
one participant (390 sex-marker rows across 17 participants) using
participant-block PSIS. That participant's complete 18S and piperaquine course
remains conditioned upon. No sex-marker observation from the held-out
participant is retained for calibration. Scores average rows within each
participant/sex, weight the two marker streams equally, and then weight
participants equally.

| Quantity | Global-allocation comparator | Hierarchical-allocation extension |
|---|---:|---:|
| Primary NLPD | 4.65571 | **4.63580** |
| Maximum Pareto-k | 0.292 | 0.318 |
| Divergences / maximum-depth hits | 0 / 0 | 0 / 0 |
| Maximum R-hat | 1.0037 | 1.0056 |
| Minimum bulk / tail ESS | 1,484 / 2,770 | 759 / 1,339 |

The comparator-minus-extension gain is **0.01991 nat per observation**
(participant-bootstrap 95% interval 0.01173 to 0.02861), with the extension
better for 15 of 17 participants.
The gain is 0.00616 for the female marker and 0.03366 for the male marker. The
secondary joint-trajectory gain is 0.01378 (95% interval 0.00668 to 0.02121;
13/17 participant wins). All participant blocks satisfy the
sample-size-dependent PSIS threshold, and both posteriors satisfy the
predefined MCMC gate. In the hierarchical-allocation extension, the population
allocation median is 0.244 (95%
interval 0.189 to 0.306), participant posterior means range from 0.191 to
0.359, and the posterior mean for `tau_rho` is 0.387 (95% interval 0.062 to
0.727).

The supported claim is deliberately narrow: the hierarchical-allocation
extension improves internal
participant-block sex-marker prediction conditional on complete participant
18S/PK relative to the matched Cao-derived global-allocation comparator. This
is not a comparison with Cao's published fit unchanged, external or prospective
validation, or causal identification of sex allocation.

In the hierarchical-allocation extension, the deleted block is also the block
that informs that participant's allocation random effect in the full posterior.
PSIS performs the required
importance deletion; the prespecified fallback is an exact `sex_include=0`
masked refit for any high-k participant. No block triggered that fallback.

## Biological interpretation

`rho_i` is a persistent, marker-informed effective male contribution, not a
direct measurement of the committed-cell sex ratio. With zero initial sex
stocks and identical linear branch kinetics, it implies a time-constant
male:female contrast within each participant. The same normalized contrast
can be moved from the immature source to the terminal assay mapping without
changing the likelihood, so these data do not identify where or when sex
determination occurs. Sex-specific transcript yield, assay recovery, and
unmodelled kinetics remain alternative explanations.

The extension is nevertheless biologically coherent as a predictive model:
it conserves the gametocyte source, retains the established within-host
backbone, and uses partial pooling rather than fitting 17 unrelated ratios.

## Reproduce

Create the environment and install the exact dependency versions:

```bash
cd sex-cao-model
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m cmdstanpy.install_cmdstan \
  --version 2.39.0 --cores 4 --dir .cmdstan
```

On a 48-core host, reproduce the two arms concurrently. Each arm uses four
chains with six threads per chain and fresh `diag_e` adaptation:

```bash
.venv/bin/python fit.py --arm b0 --output outputs/qualified_refit &
b0_pid=$!
.venv/bin/python fit.py --arm b2 --output outputs/qualified_refit &
b2_pid=$!
wait "$b0_pid" "$b2_pid"

.venv/bin/python score.py \
  --b0 outputs/qualified_refit/b0 \
  --b2 outputs/qualified_refit/b2 \
  --output outputs/qualified_score.json
```

The global-allocation comparator uses seed 2026080271, 2,500 warmup iterations,
3,000 retained draws, and `adapt_delta=0.99`. The hierarchical-allocation
extension uses seed 20250802, 2,000 warmup iterations, 2,000 retained draws,
and `adapt_delta=0.95`. Both use four chains,
`max_treedepth=13`, 18-digit CSV output, and the exact bundled chain
initializations: one complete observed ridge draw plus three dispersed starts.
The participant bootstrap uses seed 2026080370 and 50,000 replicates.

`score.py` checks the sampler configuration, matched comparator/extension data,
MCMC gate, and PSIS gate before allowing a performance result. The historical
posterior CSVs are not included; `fit.py` regenerates locally scoreable CSVs.

## Data and attribution

The bundled data contain participant-coded longitudinal measurements derived
from the public 17-volunteer Cao 3D7 release. The original dataset is Pengxing
Cao (2019), *Code for Modelling Gametocyte Population Dynamics*, University of
Melbourne, DOI
[`10.26188/5cde4c26c8201`](https://doi.org/10.26188/5cde4c26c8201).

## Package map

- `model/cao_sex_allocation.stan`: shared comparator/extension Stan program.
- `results/qualified_run/`: analysis configuration, inputs, and reported summaries.
- `data/stan_data.json`: convenience mirror of the qualified extension data used
  only by trajectory-generation tooling; refits use the per-arm data in
  `results/qualified_run/`.
- `fit.py`: exact qualified refit runner.
- `score.py`: all-marker participant-block PSIS comparison and diagnostic gate.
- `results/reported_summary.json`: concise qualified result.
- `figure_data/`, `build_figures.py`: figure inputs and rendering.
- `prepare_trajectory_summary.py`: regenerate trajectory summaries from local fits.
