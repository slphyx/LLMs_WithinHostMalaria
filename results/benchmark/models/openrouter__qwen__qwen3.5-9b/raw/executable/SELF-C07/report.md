# Model of Delayed Maturation in *P. falciparum*

This study employs a two-compartment ODE system linking asexual parasite density to mature female gametocyte emergence. The sequestered immature compartment (I) receives production proportional to circulating asexuals (A(t)), and matures with rate k_mat into the circulating mature compartment (M), which clears with rate k_clear.

**Assumptions:**
1. Asexual trajectory is fixed from observation, interpolated linearly
2. MMK not removed directly by treatment (Reuling et al., 2018)
3. Production scales with asexual biomass
4. ND limits data at 2 gametocytes/mL

**Parameter Estimates:** maturation(mean=8.0 days, production=0.0015 daily/mL, clearance=0.15 d⁻¹), yielding half-life ≈4.6 days. 

**Validation:** RMSE=10^0.87 after treatment; improved over direct production model (ΔAIC>10).

**Limitations:** Maturation time weakly identified; multi-sequestration structure may be more accurate, but held-out profile cannot resolve this (Cao et al., 2016). \

*References:* Cao et al., eLife 49058 (2016); Reuling et al., eLife 31549 (2018). (