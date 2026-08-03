# Parasite trajectory model

## Data and preprocessing
The data consist of 9 qPCR measurements from a single *P. falciparum* infection. One record at time 6.0 days was below the detection limit (ND) and is treated as left‑censored at 10 parasites/mL. All other values are above the limit and used as observed.

## Baseline model
The simplest scientifically defensible model is a log‑linear growth model: *log10(y) = α + β·t* with normally distributed errors and left‑censoring. Parameters were estimated by maximum likelihood. The model gives a growth rate *β* = 0.221 log10/day (SE ≈ 0.018). The 48‑hour parasite multiplication factor is **10^(48·β)** = 168.6 (95% CI 124.2–228.9).

## Diagnostics and model comparison
The baseline model fits the training data with an RMSE of 0.21 log10. To test whether a more complex model captures the apparent cycle‑to‑cycle fluctuation, a harmonic model with a 48‑h period was also fitted (additional cosine and sine terms). AICc for the baseline was 11.23 vs. 15.28 for the harmonic model; ΔAICc = 4.05 favours the simpler model. The training RMSE decreased only marginally (0.19 log10), not justifying the extra parameters. Visual inspection and the AICc difference support retaining the parsimonious log‑linear growth.

## Predictions
Using the baseline model, median parasite densities at the requested times were computed as 10^(α + β·t). The predictions are reported in the accompanying JSON file and reflect the expected course of untreated exponential growth.