# Parasite Growth Model Analysis

## Model Summary

We fit a Tobit regression model to left-censored parasite density data. The model assumes:
- Logarithmic growth: log₁₀(parasites) ~ Normal(β₀ + β₁ * time, σ²)
- Left-censoring at detection limit (10 parasites/ml)
- Log-linear growth rate 

**Key findings:**
- Estimated daily log₁₀ growth rate: 0.3998 [95% CI: 0.31, 0.49]
- Parasite multiplication factor (48h): 6.30 [95% CI: 4.12, 9.64]
- Training RMSE (log₀scale): 0.2237

## Biological Interpretation

The parasite multiplication factor of ~6 per 48-hour cycle indicates substantial growth. For untreated P. falciparum infections, this aligns with expected growth rates.