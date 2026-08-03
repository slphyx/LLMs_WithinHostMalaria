#!/usr/bin/env python3
"""Fit delayed female-gametocyte ODE model and regenerate all outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import log_ndtr

ROOT = Path(__file__).resolve().parent
RTOL, ATOL, LOD = 1e-8, 1e-10, 2.0
N_MAIN = 5

driver = pd.read_csv(ROOT / "input/asexual_driver.csv")
obs = pd.read_csv(ROOT / "input/gametocytes.csv")
pred_times = pd.read_csv(ROOT / "input/prediction_times.csv")["time_days"].to_numpy(float)
ta = driver["time_days"].to_numpy(float)
aa = driver["asexual_parasites_per_ml"].to_numpy(float)
tobs = obs["time_days"].to_numpy(float)
detected = obs["status"].eq("detected").to_numpy()
yobs = pd.to_numeric(obs["gametocytes_per_ml"], errors="coerce").to_numpy(float)


def simulate(theta, n_immature, times):
    """theta=(log production, log mean maturation, log clearance, log sigma)."""
    production, mean_time, clearance, sigma = np.exp(theta)
    rate = n_immature / mean_time
    times = np.asarray(times, float)
    end = float(np.max(times))

    def rhs(t, state):
        # Piecewise-linear fixed driver; endpoint values outside its domain.
        asexual = np.interp(t, ta, aa)
        dy = np.empty(n_immature + 1)
        dy[0] = production * asexual - rate * state[0]
        for j in range(1, n_immature):
            dy[j] = rate * (state[j - 1] - state[j])
        dy[-1] = rate * state[n_immature - 1] - clearance * state[-1]
        return dy

    grid = np.unique(np.r_[times, ta[(ta >= 0) & (ta <= end)]])
    sol = solve_ivp(rhs, (0.0, end), np.zeros(n_immature + 1), method="DOP853",
                    t_eval=grid, rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(sol.message)
    mature = np.interp(times, sol.t, sol.y[-1])
    return np.maximum(mature, 1e-300), sigma


def nll(theta, n_immature):
    pred, sigma = simulate(theta, n_immature, tobs)
    logp = np.log(pred)
    z = (np.log(yobs[detected]) - logp[detected]) / sigma
    # Lognormal density for detections; ND contributes P(Y < LOD).
    value = np.sum(np.log(yobs[detected] * sigma * np.sqrt(2*np.pi)) + .5*z*z)
    value -= np.sum(log_ndtr((np.log(LOD) - logp[~detected]) / sigma))
    return float(value)


def fit_model(n_immature):
    bounds = [(-12, 0), (np.log(.5), np.log(30)), (np.log(.002), np.log(2)),
              (np.log(.03), np.log(2))]
    starts = [[.01, 5, .10, .2], [.03, 9, .15, .25], [.005, 4, .03, .15]]
    best = None
    for start in starts:
        res = minimize(nll, np.log(start), args=(n_immature,), method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 700, "ftol": 1e-11, "gtol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    if not np.isfinite(best.fun):
        raise RuntimeError("optimisation did not find a finite likelihood")
    return best


main = fit_model(N_MAIN)
alternative = fit_model(1)
production, mean_time, clearance, sigma = np.exp(main.x)
main_pred, _ = simulate(main.x, N_MAIN, tobs)
rmse = float(np.sqrt(np.mean((np.log10(yobs[detected]) - np.log10(main_pred[detected]))**2)))

# Local observed-information interval on log clearance. Central finite differences
# include covariance with production, maturation time and observation noise.
p = len(main.x)
h = 2e-3
H = np.empty((p, p))
f0 = nll(main.x, N_MAIN)
for i in range(p):
    ei = np.zeros(p); ei[i] = h
    H[i, i] = (nll(main.x + ei, N_MAIN) - 2*f0 + nll(main.x - ei, N_MAIN))/h**2
    for j in range(i):
        ej = np.zeros(p); ej[j] = h
        H[i, j] = H[j, i] = (nll(main.x+ei+ej,N_MAIN)-nll(main.x+ei-ej,N_MAIN)
                                 -nll(main.x-ei+ej,N_MAIN)+nll(main.x-ei-ej,N_MAIN))/(4*h*h)
cov = np.linalg.pinv(H)
se_log_mu = float(np.sqrt(max(cov[2, 2], 0)))
mu_lo, mu_hi = np.exp(main.x[2] + np.array([-1, 1])*1.96*se_log_mu)
half_life = float(np.log(2)/clearance)
half_interval = [float(np.log(2)/mu_hi), float(np.log(2)/mu_lo)]

traj_t = np.round(np.arange(0, 26.0001, .1), 10)
traj_y, _ = simulate(main.x, N_MAIN, traj_t)
pd.DataFrame({"time_days": traj_t,
              "mature_gametocytes_per_ml": traj_y}).to_csv(ROOT / "trajectory.csv", index=False)
held, _ = simulate(main.x, N_MAIN, pred_times)

main_aic = 2*4 + 2*main.fun
alt_aic = 2*4 + 2*alternative.fun
result = {
  "model": {
    "name": "five-stage Erlang immature delay plus circulating mature compartment",
    "equations": "I1'=p A(t)-k I1; Ij'=k(I[j-1]-Ij), j=2..5; M'=k I5-mu M; k=5/T. Densities are per mL, rates per day, and all states start at zero on day 0.",
    "driver_interpolation": "piecewise-linear interpolation of supplied asexual densities (fixed input)",
    "solver": f"SciPy solve_ivp DOP853 adaptive solver; rtol={RTOL:g}, atol={ATOL:g}",
    "observation_model": "independent lognormal measurements with fitted log-SD; ND is left-censored at 2 gametocytes/mL via its lognormal CDF"
  },
  "parameters": {
    "effective_production_per_day": float(production),
    "mean_maturation_days": float(mean_time),
    "mature_clearance_rate_per_day": float(clearance),
    "mature_half_life_days": half_life,
    "mature_half_life_interval": half_interval
  },
  "predictions": [{"time_days": float(t), "gametocytes_per_ml": float(v)} for t,v in zip(pred_times, held)],
  "diagnostics": {
    "training_log10_rmse": rmse,
    "alternative_model": f"One exponential immature compartment (same 4 fitted parameters) had AIC {alt_aic:.2f}, versus {main_aic:.2f} for the five-stage Erlang model (Delta AIC {alt_aic-main_aic:.2f}); its training log10 RMSE was {np.sqrt(np.mean((np.log10(yobs[detected])-np.log10(simulate(alternative.x,1,tobs)[0][detected]))**2)):.3f}.",
    "identifiability": "In this single infection, production, maturation and clearance trade off. The half-life interval uses local observed information and conditions on five stages and the fixed driver; maturation-stage count is structural, not estimated. No direct treatment effect on immature or mature states was included.",
    "fitted_log_sd": float(sigma),
    "clearance_log_scale_standard_error": se_log_mu
  }
}
(ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")

report = f"""# Delayed female-gametocyte model

I treated the supplied asexual density, *A(t)*, as fixed and piecewise-linearly interpolated. The selected non-negative transit model has five sequestered immature compartments: `I1'=pA-kI1`, `Ij'=k(I[j-1]-Ij)` (j=2,...,5), and circulating mature females `M'=kI5-mu M`, with `k=5/T`. States were zero at day 0. Treatment therefore acts only through the supplied fall in *A(t)*, with no additional gametocytocidal term. Multiple transit compartments approximate a distributed developmental delay. Immature stages I-IV are known to sequester before mature stage V returns to blood; Cao et al. estimated about 8.4 days of sequestration [1].

Detections used an independent lognormal likelihood with fitted log-SD {sigma:.3f}; ND contributed the probability below 2/mL rather than being set to zero. DOP853 adaptive integration used relative tolerance {RTOL:g} and absolute tolerance {ATOL:g}. Maximum likelihood estimates were effective production **{production:.5g}/day**, mean maturation **{mean_time:.2f} days**, and mature clearance **{clearance:.4f}/day**. Thus mature half-life was **{half_life:.2f} days** (approximate 95% observed-information interval **{half_interval[0]:.2f}-{half_interval[1]:.2f} days**). Reuling et al. reported a female Pfs25-based half-life of 5.1 days (95% CI 4.1-6.1), providing biological context rather than external fitting data [2].

Training RMSE was {rmse:.3f} log10 units. A one-compartment exponential maturation alternative had AIC {alt_aic:.1f}, versus {main_aic:.1f} for the five-stage model (Delta={alt_aic-main_aic:.1f}), supporting a narrower delay. Requested predictions are in `results.json`; the dense fitted curve is in `trajectory.csv`.

This single short profile weakly separates production, delay shape, and mature loss: continued production from sequestered cohorts can mimic slow clearance. The interval conditions on five stages and the fixed driver, so it excludes structural/driver uncertainty. Assay RNA may also not equal viable-cell abundance.

## References
1. Cao et al. (2019), *eLife* 8:e49058. https://doi.org/10.7554/eLife.49058
2. Reuling et al. (2018), *eLife* 7:e31549. https://doi.org/10.7554/eLife.31549
"""
(ROOT / "report.md").write_text(report)
print(json.dumps({"nll": main.fun, "parameters": result["parameters"], "runtime_outputs": ["results.json", "trajectory.csv", "report.md"]}, indent=2))
