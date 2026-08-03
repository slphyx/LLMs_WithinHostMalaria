I fitted a non-negative delayed-maturation ODE with a fixed asexual driver A(t) and three sequestered immature compartments I1–I3 plus circulating mature gametocytes M:

- dI1/dt = βA(t) − 3kI1
- dIi/dt = 3k(Ii−1 − Ii), i = 2,3
- dM/dt = 3kI3 − cM

A(t) was interpolated by piecewise-linear interpolation on the supplied arithmetic-scale asexual trajectory and held constant outside the supplied range. Treatment was represented only through the fixed asexual driver; it had no direct effect on gametocytes. Observed gametocyte values were modeled as lognormal with left-censoring at 2 gametocytes/mL for ND. I used SciPy’s adaptive LSODA solver with rtol = 1e-7 and atol = 1e-10.

The fitted effective production coefficient was 5.22e-3, the mean immature maturation time was 3.80 days, and the mature clearance rate was 1.0e-4/day, implying a mature half-life of 6931 days (95% interval 976–49209 days). The mature half-life is weakly identified because the profile contains only one ND value and no clear post-peak decay over the observation window.

Compared with a one-immature-compartment alternative, the 3-compartment Erlang structure fit much better (AIC 172.1 vs 1167.1), supporting delayed appearance through sequestration rather than immediate release.

Biological assumptions are consistent with prior work showing sequestered immature stages before circulating stage V and delayed detectable gametocytemia in controlled human infection (Cao et al., 2019; Reuling et al., 2018). The exact number of immature compartments is not identifiable from this profile, so the chain length is a modeling convenience rather than a measured stage count.

Citations: Cao et al., eLife 2019, doi:10.7554/eLife.49058; Reuling et al., eLife 2018, doi:10.7554/eLife.31549.