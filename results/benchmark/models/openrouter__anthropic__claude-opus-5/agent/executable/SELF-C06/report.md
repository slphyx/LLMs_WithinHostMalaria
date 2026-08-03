# Fitting an untreated *P. falciparum* parasitaemia trajectory

## Data and observation model
Nine pre-treatment qPCR samples (t = 6–11 d). The t = 6 d non-detect is treated
as left-censored at the detection limit of 10 parasites/mL, not as zero: it
contributes Φ((log10 10 − μ)/σ) to the likelihood (Tobit). All other points
contribute a Gaussian density for log10 density with common σ. Multiplicative
qPCR error makes the log10 scale the natural one.

## Models
*Baseline*: free-intercept log-linear growth, log10 N(t) = a + b·t.
*Extension*: the same trend plus a 48-h sinusoid, + c·cos(πt) + d·sin(πt),
motivated by synchronous schizogony and sequestration, which make circulating
density oscillate over one cycle without changing the mean growth rate.

Residuals from the baseline are not random: samples at even days lie ~0.35
log10 below trend and odd days ~0.2 above, a systematic 2-day pattern. The
extension lowers AICc (9.70 vs 10.69), cuts leave-one-out log10-RMSE from 0.259
to 0.119, and reduces σ from 0.22 to 0.068 log10. Both criteria favour it, so
the extra two parameters are accepted; otherwise the baseline would be kept.

## Results
Trend slope b = 0.382 log10/day, i.e. a multiplication factor of
**5.8 per 48 h (95% CI 4.7–7.1)** — typical of untreated early infection
(literature PMF ≈ 3–12). The oscillation amplitude is 0.31 log10 (~2-fold
peak-to-trough), consistent with cyclical sequestration.

Predicted densities (parasites/mL): 22 (7.5 d), 128 (9.5 d), 741 (11.5 d),
933 (12.0 d); 95% predictive intervals are in `results.json`.

## Caveats
Only 8 detected points, so the cycle phase rests on few observations; the
half-day predictions fall at a cycle phase not directly sampled, and the
baseline model would predict ~1.7× higher there. Growth is assumed
density-independent over this short, sub-clinical window.
