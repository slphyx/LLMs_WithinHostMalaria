I fit a left-censored log-linear model to log10 parasitaemia:

log10(P(t)) = a + bt + ε,  ε ~ Normal(0, σ²)

The ND measurement was treated as left-censored at 10 parasites/mL, not as zero. The fitted slope was b = 0.400 log10/mL per day, implying an average 48-hour multiplication factor of 6.30 (95% interval 4.68–10.19). The fitted trajectory predicts densities of 36.6, 231, 1456, and 2307 parasites/mL at 7.5, 9.5, 11.5, and 12.0 days.

I compared this baseline to a quadratic time trend. The curvature term did not improve AIC and added no clear biological value, so I retained the simpler free-intercept log-linear model. Residual scatter on the detected points was modest (training RMSE 0.224 log10 units), supporting the adequacy of a constant-growth description over the observed window.
