"""SELF-C07: delayed appearance of circulating mature female gametocytes.

Reads only input/*.csv. Fits a linear-chain (gamma) maturation ODE model
driven by a fixed interpolated asexual trajectory.
"""
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm, chi2

RNG = np.random.default_rng(20240607)
LOD = 2.0            # gametocytes/mL, left-censoring limit
RTOL, ATOL = 1e-7, 1e-9

drv = pd.read_csv("input/asexual_driver.csv")
gam = pd.read_csv("input/gametocytes.csv")
pred_t = pd.read_csv("input/prediction_times.csv")["time_days"].to_numpy(float)

# Fixed driver: piecewise-linear interpolation of asexual density on the
# natural scale (preserves the zeros after treatment), constant extrapolation.
_at = drv.time_days.to_numpy(float)
_av = drv.asexual_parasites_per_ml.to_numpy(float)


def A(t):
    return np.interp(t, _at, _av, left=_av[0], right=0.0)

t_obs = gam.time_days.to_numpy(float)
detected = (gam.status == "detected").to_numpy()
y_obs = np.where(detected, pd.to_numeric(gam.gametocytes_per_ml,
                                         errors="coerce").fillna(LOD), LOD)


def simulate(theta, n, t_eval):
    """theta = (log f, log tau, log delta). States: I_1..I_n (sequestered), G."""
    f, tau, delta = np.exp(theta)
    k = n / tau

    def rhs(t, y):
        d = np.empty(n + 1)
        d[0] = f * max(float(A(t)), 0.0) - k * y[0]
        if n > 1:
            d[1:n] = k * (y[0:n - 1] - y[1:n])
        d[n] = k * y[n - 1] - delta * y[n]
        return d

    t_max = max(t_eval.max(), 26.0)
    sol = solve_ivp(rhs, (0.0, t_max), np.zeros(n + 1), method="LSODA",
                    t_eval=np.unique(np.concatenate([[0.0], t_eval])),
                    rtol=RTOL, atol=ATOL, dense_output=True)
    if not sol.success:
        return None, None
    g = np.clip(sol.sol(t_eval)[n], 0.0, None)
    return g, sol


def nll(p, n):
    theta, log_sigma = p[:3], p[3]
    sigma = np.exp(log_sigma)
    g, _ = simulate(theta, n, t_obs)
    if g is None:
        return 1e10
    mu = np.log10(np.clip(g, 1e-12, None))
    ll = 0.0
    ll += norm.logpdf(np.log10(y_obs[detected]), mu[detected], sigma).sum()
    ll += norm.logcdf((np.log10(LOD) - mu[~detected]) / sigma).sum()
    if not np.isfinite(ll):
        return 1e10
    return -ll


def fit(n):
    best = None
    for f0, tau0, d0 in ((1e-2, 9.0, 0.2), (1e-1, 5.0, 0.1)):
        p0 = np.array([np.log(f0), np.log(tau0), np.log(d0), np.log(0.2)])
        r = minimize(nll, p0, args=(n,), method="Nelder-Mead",
                     options=dict(maxfev=700, xatol=1e-4, fatol=1e-6))
        r = minimize(nll, r.x, args=(n,), method="Nelder-Mead",
                     options=dict(maxfev=700, xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best


models = {}
for n in (1, 2, 3, 5):
    r = fit(n)
    k_par = 4
    models[n] = dict(res=r, nll=r.fun, aic=2 * r.fun + 2 * k_par)
    print(n, r.fun, np.exp(r.x))

n_best = min(models, key=lambda n: models[n]["aic"])
res = models[n_best]["res"]
f_h, tau_h, delta_h, sigma_h = np.exp(res.x)

# Profile-likelihood 95% CI for delta -> mature half-life
crit = chi2.ppf(0.95, 1) / 2


def prof_nll(log_delta):
    def obj(q):
        return nll(np.array([q[0], q[1], log_delta, q[2]]), n_best)
    r = minimize(obj, np.array([res.x[0], res.x[1], res.x[3]]),
                 method="Nelder-Mead", options=dict(maxfev=400, xatol=1e-3, fatol=1e-5))
    return r.fun


def find_bound(direction):
    lo, hi = res.x[2], res.x[2]
    step = 0.15 * direction
    for _ in range(40):
        hi = hi + step
        if prof_nll(hi) - res.fun > crit:
            break
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if prof_nll(mid) - res.fun > crit:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < 1e-3:
            break
    return np.exp(0.5 * (lo + hi))


d_lo, d_hi = find_bound(+1), find_bound(-1)   # delta bounds
hl = np.log(2) / delta_h
hl_ci = sorted([np.log(2) / d_lo, np.log(2) / d_hi])

# fits / predictions
g_train, _ = simulate(res.x[:3], n_best, t_obs)
resid = np.log10(np.clip(g_train[detected], 1e-12, None)) - np.log10(y_obs[detected])
rmse = float(np.sqrt(np.mean(resid ** 2)))

g_pred, _ = simulate(res.x[:3], n_best, pred_t)
t_grid = np.arange(0.0, 26.0 + 1e-9, 0.05)
g_grid, _ = simulate(res.x[:3], n_best, t_grid)
pd.DataFrame({"time_days": t_grid,
              "mature_gametocytes_per_ml": g_grid}).to_csv(
    "trajectory.csv", index=False)

alt_txt = "; ".join(
    f"n={n}: nll={models[n]['nll']:.2f}, AIC={models[n]['aic']:.2f}, "
    f"tau={np.exp(models[n]['res'].x[1]):.2f} d, "
    f"t1/2={np.log(2)/np.exp(models[n]['res'].x[2]):.2f} d"
    for n in sorted(models))

out = {
    "model": {
        "name": f"Linear-chain (Erlang) gametocyte maturation model, n={n_best} "
                "sequestered immature compartments",
        "equations": (
            "dI1/dt = f*A(t) - (n/tau)*I1; dIi/dt = (n/tau)*I(i-1) - (n/tau)*Ii "
            "(i=2..n); dG/dt = (n/tau)*In - delta*G. "
            "I_i, G in gametocytes/mL; A(t) in asexual parasites/mL; "
            "f in (gametocytes/mL)/(asexual parasites/mL)/day; tau in days; "
            "delta per day. All states start at 0 and stay non-negative."),
        "driver_interpolation": (
            "A(t) piecewise-linear interpolation of the supplied asexual values "
            "on the natural (non-log) scale, clipped at >=0; constant "
            "extrapolation (first value before t=0, 0 after t=26)."),
        "solver": f"scipy solve_ivp LSODA (adaptive, stiff-capable), "
                  f"rtol={RTOL}, atol={ATOL}",
        "observation_model": (
            "log10-normal measurement error: log10(y) ~ Normal(log10(G(t)), "
            "sigma); ND values contribute the left-censored probability "
            "Phi((log10(2)-log10(G))/sigma) at LOD = 2 gametocytes/mL. "
            f"MLE fit; sigma_hat = {sigma_h:.3f} log10 units."),
    },
    "parameters": {
        "effective_production_per_day": float(f_h),
        "mean_maturation_days": float(tau_h),
        "mature_clearance_rate_per_day": float(delta_h),
        "mature_half_life_days": float(hl),
        "mature_half_life_interval": [float(hl_ci[0]), float(hl_ci[1])],
    },
    "predictions": [{"time_days": float(t), "gametocytes_per_ml": float(g)}
                    for t, g in zip(pred_t, g_pred)],
    "diagnostics": {
        "training_log10_rmse": rmse,
        "alternative_model": (
            "Compared Erlang chains with n=1 (single immature compartment, "
            "exponential maturation delay), n=2, n=3 and n=5 (more "
            "deterministic delay) by profile MLE and AIC: " + alt_txt +
            f". Selected n={n_best} by lowest AIC."),
        "identifiability": (
            "f and delta are only weakly separated: the mature plateau depends "
            "on f/delta, so a compensating increase in both is nearly "
            "equivalent; the profile CI for the mature half-life is therefore "
            "wide. tau trades off against the asexual peak timing and is only "
            "constrained by the early rise (one ND point at day 5), so the "
            "number of immature compartments (delay shape/variance) is poorly "
            "resolved by a single profile. No post-day-20 data means the "
            "long-term decay of mature gametocytes after asexual clearance "
            "(hence delta) is extrapolated."),
    },
}
with open("results.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out["parameters"], indent=2))
print(out["predictions"])
