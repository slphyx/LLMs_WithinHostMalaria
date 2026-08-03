# Untreated parasite trajectory

A free-intercept log-linear growth model was fitted on the log10 parasites/mL scale. Detected observations had independent Gaussian residuals with an estimated standard deviation of 0.222 log10 units; the ND was included through its left-censored likelihood below 10 parasites/mL, rather than treated as zero or 10.

The estimated growth rate was 0.400 log10/day, corresponding to a **48-hour parasite multiplication factor (PMF) of 6.30**. The 95% profile-likelihood interval is **4.02–10.50**. This describes average net peripheral-density growth, not an independently identified replication mechanism.

A more complex candidate added sine and cosine terms with the known 48-hour blood-stage period. Although it captured alternating residual structure, leave-one-out performance did not support it: detected-point RMSE was 0.126 versus 0.275 log10, but total censored predictive log score was -16.97 versus -4.45. Its apparent precision was unstable when individual points were omitted. The parsimonious baseline was therefore retained. Training RMSE (detected values) was 0.224 log10.

Predictions are median latent parasite densities from the fitted mean curve. They extrapolate only one day beyond the observed range at most; uncertainty grows outside that range and no treatment effect is represented.
