# Untreated *Plasmodium falciparum* Parasitaemia Trajectory Fit

## Model Formulation & Statistical Likelihood
Pre-treatment qPCR measurements exhibit exponential growth confounded by periodic synchronous schizogony/sequestration. We formulated a log-linear model with a 48-hour harmonic component:
$$\log_{10}(P(t)) = a + b \cdot t + A \sin(\pi t) + B \cos(\pi t)$$
The observation process accounts for left-censoring below the qPCR detection limit ($L = 10 \text{ parasites/mL}$, $\log_{10} L = 1.0$) via a Tobit Gaussian likelihood:
$$\mathcal{L}(\theta) = \prod_{i: \text{det}} \frac{1}{\sigma}\phi\left(\frac{y_i - \mu(t_i)}{\sigma}\right) \prod_{j: \text{cens}} \Phi\left(\frac{\log_{10}L - \mu(t_j)}{\sigma}\right)$$

## Diagnostic & Predictive Evidence
Compared to the free-intercept log-linear baseline ($a + b t$), the 48-hour harmonic model dramatically improves fit quality across all criteria:
- **Training $\log_{10}$ RMSE:** Reduced from $0.2237$ to $0.0680$.
- **Model Selection (AIC):** Improved from $5.89$ (baseline) to $-10.30$ (harmonic).
- **Out-of-Sample LOOCV RMSE:** Reduced from $0.2746$ to $0.1259$, confirming strong predictive generalization without overfitting.

Optimizing the oscillation period directly yields $T = 2.002 \text{ days}$, confirming the biological 48-hour cycle of *P. falciparum*.

## Parasite Multiplication Factor (PMF)
The log10 growth rate per day is $b = 0.3821$. Over a standard 48-hour (2-day) cycle, the parasite multiplication factor (PMF) is:
$$\text{PMF}_{48\text{h}} = 10^{2b} = 5.81$$
Profile likelihood ratio bounds ($2\Delta \text{NLL} = 3.841$) yield a 95% confidence interval of **[4.93, 6.85]**, consistent with typical *in vivo* net expansion rates.

## Predictions
Expected peripheral parasitaemia at held-out times:
- $t = 7.5\text{ d}$: $21.96 \text{ parasites/mL}$
- $t = 9.5\text{ d}$: $127.57 \text{ parasites/mL}$
- $t = 11.5\text{ d}$: $741.13 \text{ parasites/mL}$
- $t = 12.0\text{ d}$: $932.87 \text{ parasites/mL}$
