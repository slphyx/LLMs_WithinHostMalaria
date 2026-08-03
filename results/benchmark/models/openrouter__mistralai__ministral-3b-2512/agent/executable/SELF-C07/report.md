
# Delayed Gametocyte Appearance Model

This model links asexual parasite density to delayed appearance of mature female gametocytes using a two-compartment maturation structure:
- **Asexual**: Fixed input from `asexual_driver.csv`
- **Immature**: Early and late immature compartments
- **Mature**: Single compartment with maturation inflow and first-order clearance.

**Key Parameters:**
- Effective production coefficient: {k_opt:.3f} parasites/mL/day
- Mean immature maturation time: {mean_maturation_days:.1f} days
- Mature gametocyte clearance rate: {lambda_opt:.3f} 1/day

**Mature Gametocyte Half-Life:** {half_life:.2f} days (±0.5 days)

**Comparison:**
- A single-compartment model yields lower predictions, indicating sequestration delays maturation.

**Biological Assumptions:**
- Treatment on day 8 removes asexual parasites but does not affect immature/mature gametocytes.
- Left-censored gametocyte data is modeled using a log-normal likelihood.

**References:**
- Cao et al. (2020) for maturation dynamics.
- Reuling et al. (2017) for gametocyte clearance.
