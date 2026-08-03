# Parasite trajectory analysis: SELF-C06

## Data

Nine qPCR measurements from a single *P. falciparum* infection spanning
6.0-11.0 days. One observation (t = 6.0 d) was below the 10 parasites/mL
detection limit and is treated as left-censored. All other measurements
range from 20 to 1106 parasites/mL.

## Model

The selected model is a free-intercept log-linear growth model on the
log10 scale:

> log10(parasites/mL) = a + b * time_days

with independent Normal(0, sigma^2) observation errors. Measurements
below the detection limit contribute Phi((log10(10) - mu) / sigma) to
the likelihood. Parameters were estimated by maximum likelihood using
scipy.optimize, with standard errors from the observed Fisher
information.

A logistic growth model (adding a carrying capacity K and half-maximum
time t_half) was also fitted. Delta_AICc = +7.2 favours the log-linear
model; the logistic model pushes K to ~9e8 parasites/mL with a t_half
far beyond the data range (26 d), confirming it is not identifiable.

## Results

| Quantity                    | Value                |
|-----------------------------|----------------------|
| log10 growth rate (per day) | 0.400 (SE 0.047)     |
| PMF per 48 h                | 6.30                 |
| 95 % CI for PMF             | [4.13, 9.61]         |
| Residual sigma (log10)      | 0.222                |
| Training RMSE (log10)       | 0.224                |

Residual diagnostics show no systematic departure from log-linearity
(lag-1 autocorrelation = -0.46, Shapiro-Wilk p = 0.13).

## Predictions

| Time (days) | Predicted parasites/mL |
|-------------|------------------------|
| 7.5         | 37                     |
| 9.5         | 231                    |
| 11.5        | 1456                   |
| 12.0        | 2307                   |

## Interpretation

The estimated PMF of 6.3 per 48-hour cycle (95 % CI 4.1-9.6) is
consistent with moderate *P. falciparum* multiplication. The log-linear
model is the simplest defensible choice: residuals are compatible with
normality and show no curvature, and the more complex logistic model
gains no predictive support.