"""SELF-C06: fit an untreated P. falciparum parasitaemia trajectory.

Baseline: free-intercept log-linear (exponential) growth with a Tobit
(left-censored) Gaussian likelihood on log10 density.
Candidate extension: same growth plus a 48-h periodic term (sequestration /
synchronous cycle), fitted by MLE and compared by AICc and leave-one-out CV.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

LOD = 10.0
LOG_LOD = np.log10(LOD)
RNG = np.random.default_rng(20240606)


def load():
    d = pd.read_csv("input/parasitaemia.csv")
    cens = d["status"].str.contains("below").values
    y = np.where(cens, LOG_LOD, pd.to_numeric(d["parasites_per_ml"], errors="coerce").fillna(LOD))
    return d["time_days"].values.astype(float), np.log10(np.asarray(y, float)), cens


def design(t, periodic):
    cols = [np.ones_like(t), t]
    if periodic:
        cols += [np.cos(np.pi * t), np.sin(np.pi * t)]  # period = 2 days
    return np.column_stack(cols)


def negll(par, X, y, cens):
    beta, ls = par[:-1], par[-1]
    s = np.exp(ls)
    mu = X @ beta
    ll = np.where(cens,
                  norm.logcdf((LOG_LOD - mu) / s),
                  norm.logpdf(y, mu, s))
    return -ll.sum()


def fit(t, y, cens, periodic):
    X = design(t, periodic)
    p0 = np.r_[np.linalg.lstsq(X, y, rcond=None)[0], np.log(0.3)]
    res = minimize(negll, p0, args=(X, y, cens), method="Nelder-Mead",
                   options={"maxiter": 40000, "maxfev": 40000, "xatol": 1e-10, "fatol": 1e-10})
    res = minimize(negll, res.x, args=(X, y, cens), method="BFGS")
    k = len(res.x)
    n = len(y)
    aicc = 2 * res.fun + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)
    return res, aicc, X


def loo_rmse(t, y, cens, periodic):
    errs = []
    for i in range(len(t)):
        m = np.ones(len(t), bool)
        m[i] = False
        r, _, _ = fit(t[m], y[m], cens[m], periodic)
        mu = design(t[i:i + 1], periodic) @ r.x[:-1]
        if cens[i]:
            if mu[0] > LOG_LOD:
                errs.append((mu[0] - LOG_LOD) ** 2)  # penalise only over-prediction
            else:
                errs.append(0.0)
        else:
            errs.append((mu[0] - y[i]) ** 2)
    return float(np.sqrt(np.mean(errs)))


def main():
    t, y, cens = load()
    fits = {}
    for name, per in (("log-linear", False), ("log-linear + 48h periodic", True)):
        res, aicc, X = fit(t, y, cens, per)
        fits[name] = dict(res=res, aicc=aicc, X=X, periodic=per,
                          loo=loo_rmse(t, y, cens, per))

    base = fits["log-linear"]
    alt = fits["log-linear + 48h periodic"]
    # parsimony rule: keep the baseline unless the extension improves AICc and clearly improves LOO
    use_alt = (alt["aicc"] <= base["aicc"]) and (alt["loo"] < 0.8 * base["loo"])
    chosen_name = "log-linear + 48h periodic" if use_alt else "log-linear"
    ch = fits[chosen_name]
    res, X, per = ch["res"], ch["X"], ch["periodic"]
    beta = res.x[:-1]
    sigma = float(np.exp(res.x[-1]))

    # slope uncertainty: inverse observed information (central finite differences)
    def nll(p):
        return negll(p, X, y, cens)

    p = res.x
    m = len(p)
    h = 1e-4 * np.maximum(np.abs(p), 1.0)
    H = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            ei = np.zeros(m); ei[i] = h[i]
            ej = np.zeros(m); ej[j] = h[j]
            H[i, j] = (nll(p + ei + ej) - nll(p + ei - ej)
                       - nll(p - ei + ej) + nll(p - ei - ej)) / (4 * h[i] * h[j])
    cov = np.linalg.inv(H)
    se_b = float(np.sqrt(cov[1, 1]))
    dof = max(len(y) - len(res.x), 1)
    tcrit = float(student_t.ppf(0.975, dof))
    slope = float(beta[1])
    pmf = 10 ** (2 * slope)
    lo = 10 ** (2 * (slope - tcrit * se_b))
    hi = 10 ** (2 * (slope + tcrit * se_b))

    ptimes = pd.read_csv("input/prediction_times.csv")["time_days"].values.astype(float)
    Xp = design(ptimes, per)
    mu_p = Xp @ beta
    # predictive sd = parameter uncertainty + observation noise
    sd_p = np.sqrt(np.einsum("ij,jk,ik->i", Xp, cov[:-1, :-1], Xp) + sigma ** 2)
    preds = [{"time_days": float(a), "parasites_per_ml": float(10 ** b)} for a, b in zip(ptimes, mu_p)]
    pred_int = [[float(10 ** (b - tcrit * s_)), float(10 ** (b + tcrit * s_))]
                for b, s_ in zip(mu_p, sd_p)]

    mu_tr = X @ beta
    fitted_mask = ~cens
    rmse = float(np.sqrt(np.mean((mu_tr[fitted_mask] - y[fitted_mask]) ** 2)))

    out = {
        "model": {
            "name": "Free-intercept log-linear (exponential) growth, censored MLE"
                    + (" with 48-h periodic term" if per else ""),
            "formula": "log10 N(t) = a + b t" + (" + c cos(pi t) + d sin(pi t)" if per else ""),
            "observation_model": ("Gaussian residuals on log10 density with constant sigma; "
                                 "the t=6 d non-detect enters as a left-censored (Tobit) "
                                 "contribution Phi((log10(10)-mu)/sigma)."),
        },
        "growth": {
            "log10_rate_per_day": slope,
            "pmf_48h": float(pmf),
            "pmf_48h_interval": [float(lo), float(hi)],
        },
        "predictions": preds,
        "diagnostics": {
            "training_log10_rmse": rmse,
            "baseline_comparison": (
                f"Baseline log-linear: AICc={base['aicc']:.2f}, LOO log10-RMSE={base['loo']:.3f}. "
                f"48-h periodic extension: AICc={alt['aicc']:.2f}, LOO log10-RMSE={alt['loo']:.3f}. "
                f"Selected: {chosen_name}; the extension was accepted only because it improved AICc "
                f"and cut LOO log10-RMSE by >20%."),
            "sigma_log10": sigma,
            "aicc": {k: float(v["aicc"]) for k, v in fits.items()},
            "prediction_95pct_intervals": pred_int,
            "amplitude_log10_48h_cycle": float(np.hypot(beta[2], beta[3])) if per else 0.0,
            "loo_log10_rmse": {k: float(v["loo"]) for k, v in fits.items()},
        },
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
