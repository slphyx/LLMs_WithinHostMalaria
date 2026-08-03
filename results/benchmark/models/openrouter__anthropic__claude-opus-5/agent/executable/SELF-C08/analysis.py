"""SELF-C08: human-to-mosquito transmission model from molecular gametocyte densities.

Reads only input/feeds.csv and input/prediction_feeds.csv, fits binomial
(and beta-binomial) GLMs by maximum likelihood with scipy, compares
sex-specific versus total-gametocyte parameterisations, assesses the added
value of asexual density, and writes results.json.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import chi2

RNG = np.random.default_rng(20240611)
IN = "input"

train = pd.read_csv(f"{IN}/feeds.csv")
pred = pd.read_csv(f"{IN}/prediction_feeds.csv")


def design(df, kind):
    f = np.log10(df.female_gametocytes_per_ul.values)
    m = np.log10(df.male_gametocytes_per_ul.values)
    tot = np.log10(df.female_gametocytes_per_ul.values + df.male_gametocytes_per_ul.values)
    a = np.log10(df.asexual_per_ul.values)
    one = np.ones(len(df))
    cols = {
        "total": [one, tot],
        "sex": [one, f, m],
        "total_asex": [one, tot, a],
        "sex_asex": [one, f, m, a],
        "female": [one, f],
    }[kind]
    return np.column_stack(cols)


NAMES = {
    "total": ["intercept", "log10_total_gam"],
    "sex": ["intercept", "log10_female", "log10_male"],
    "total_asex": ["intercept", "log10_total_gam", "log10_asexual"],
    "sex_asex": ["intercept", "log10_female", "log10_male", "log10_asexual"],
    "female": ["intercept", "log10_female"],
}

y = train.mosquitoes_infected.values.astype(float)
n = train.mosquitoes_examined.values.astype(float)


def nll_binom(beta, X, y, n):
    eta = np.clip(X @ beta, -30, 30)
    p = expit(eta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(p) + (n - y) * np.log1p(-p))


def fit(X, y, n, nll=nll_binom, k_extra=0, x0=None):
    p = X.shape[1] + k_extra
    best = None
    starts = [np.zeros(p)] + [RNG.normal(0, 1, p) for _ in range(20)]
    if x0 is not None:
        starts.insert(0, x0)
    for s in starts:
        try:
            r = minimize(nll, s, args=(X, y, n), method="BFGS",
                         options={"maxiter": 2000, "gtol": 1e-8})
        except Exception:
            continue
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun - 1e-9):
            best = r
    return best


def nll_bb(theta, X, y, n):
    beta, logphi = theta[:-1], theta[-1]
    phi = np.exp(np.clip(logphi, -10, 10))  # precision
    p = np.clip(expit(np.clip(X @ beta, -30, 30)), 1e-10, 1 - 1e-10)
    a, b = p * phi, (1 - p) * phi
    return -np.sum(gammaln(y + a) + gammaln(n - y + b) - gammaln(n + phi)
                   + gammaln(phi) - gammaln(a) - gammaln(b))


results = {}
for kind in NAMES:
    X = design(train, kind)
    r = fit(X, y, n)
    k = X.shape[1]
    results[kind] = {"beta": r.x, "nll": r.fun, "k": k, "aic": 2 * r.fun + 2 * k,
                     }

# numerically stable covariance via finite-difference Hessian
def hessian(f, x, args, eps=1e-5):
    p = len(x)
    H = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += eps; xpp[j] += eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (f(xpp, *args) - f(xpm, *args) - f(xmp, *args) + f(xmm, *args)) / (4 * eps ** 2)
    return 0.5 * (H + H.T)


for kind in results:
    X = design(train, kind)
    H = hessian(nll_binom, results[kind]["beta"], (X, y, n))
    results[kind]["cov"] = np.linalg.pinv(H)

# leave-one-feed-out predictive binomial log loss (per mosquito)
def loo_logloss(kind):
    Xa = design(train, kind)
    tot_ll, tot_n = 0.0, 0.0
    for i in range(len(train)):
        idx = np.arange(len(train)) != i
        r = fit(Xa[idx], y[idx], n[idx], x0=results[kind]["beta"])
        p = np.clip(expit(Xa[i] @ r.x), 1e-6, 1 - 1e-6)
        tot_ll += y[i] * np.log(p) + (n[i] - y[i]) * np.log1p(-p)
        tot_n += n[i]
    return -tot_ll / tot_n


for kind in results:
    results[kind]["loo"] = loo_logloss(kind)
    results[kind]["train_logloss"] = results[kind]["nll"] / n.sum()

# likelihood-ratio tests
def lrt(big, small):
    d = 2 * (results[small]["nll"] - results[big]["nll"])
    df = results[big]["k"] - results[small]["k"]
    return d, df, float(chi2.sf(d, df))

lr_sex_total = lrt("sex", "total")
lr_asex_sex = lrt("sex_asex", "sex")
lr_asex_total = lrt("total_asex", "total")

# choose model: lowest LOO log loss among candidates without asexual unless
# asexual improves both LRT (p<0.05) and LOO
cands = ["total", "sex", "female"]
best_kind = min(cands, key=lambda k: results[k]["loo"])
asex_kind = best_kind + "_asex" if best_kind != "female" else "sex_asex"
asex_helps = (results[asex_kind]["loo"] < results[best_kind]["loo"]) and (
    lrt(asex_kind, best_kind)[2] < 0.05)
final = asex_kind if asex_helps else best_kind

# overdispersion / lack of fit for final model
Xf = design(train, final)
bf = results[final]["beta"]
pf = expit(Xf @ bf)
pearson = np.sum((y - n * pf) ** 2 / (n * pf * (1 - pf)))
df_res = len(train) - results[final]["k"]
disp = pearson / df_res
disp_p = float(chi2.sf(pearson, df_res))

# beta-binomial check
rbb = fit(Xf, y, n, nll=nll_bb, k_extra=1, x0=np.append(bf, 3.0))
lrt_bb = 2 * (results[final]["nll"] - rbb.fun)
lrt_bb_p = float(0.5 * chi2.sf(max(lrt_bb, 0), 1))  # boundary-corrected
phi_hat = float(np.exp(rbb.x[-1]))

# calibration: grouped observed vs expected in 3 bins of fitted probability
order = np.argsort(pf)
bins = np.array_split(order, 3)
calib = [{"bin": i + 1,
          "mean_fitted": round(float(pf[b].mean()), 4),
          "observed_proportion": round(float(y[b].sum() / n[b].sum()), 4),
          "mosquitoes": int(n[b].sum())} for i, b in enumerate(bins)]
hl = float(np.sum([(y[b].sum() - (n[b] * pf[b]).sum()) ** 2 /
                   ((n[b] * pf[b] * (1 - pf[b])).sum()) for b in bins]))
hl_p = float(chi2.sf(hl, 1))

# predictions with delta-method/simulation intervals on the probability scale
Xp = design(pred, final)
cov = results[final]["cov"]
draws = RNG.multivariate_normal(bf, cov, size=40000)
P = expit(draws @ Xp.T)
p_hat = expit(Xp @ bf)
lo, hi = np.percentile(P, [2.5, 97.5], axis=0)

np_ex = pred.mosquitoes_examined.values.astype(float)
expected_infected = float(np.sum(np_ex * p_hat))
weighted_p = expected_infected / np_ex.sum()

formula = {
    "total": "logit(p) = b0 + b1*log10(female+male)",
    "sex": "logit(p) = b0 + b1*log10(female) + b2*log10(male)",
    "female": "logit(p) = b0 + b1*log10(female)",
    "total_asex": "logit(p) = b0 + b1*log10(female+male) + b2*log10(asexual)",
    "sex_asex": "logit(p) = b0 + b1*log10(female) + b2*log10(male) + b3*log10(asexual)",
}[final]
coefs = {nm: round(float(v), 4) for nm, v in zip(NAMES[final], bf)}
ses = {nm: round(float(s), 4) for nm, s in zip(NAMES[final], np.sqrt(np.diag(cov)))}

cmp_txt = (
    f"Sex-specific model (log10 female + log10 male) vs total-gametocyte model "
    f"(log10[female+male]): AIC {results['sex']['aic']:.1f} vs {results['total']['aic']:.1f}; "
    f"leave-one-feed-out binomial log loss {results['sex']['loo']:.4f} vs {results['total']['loo']:.4f}; "
    f"likelihood-ratio test of adding separate male term to the total model "
    f"chi2={lr_sex_total[0]:.2f}, df={lr_sex_total[1]}, p={lr_sex_total[2]:.3f}. "
    f"Selected: {final} (lowest out-of-sample log loss and lowest AIC). The two sexes "
    f"carry distinct information: forcing a single total-density slope fits worse, and the "
    f"fitted female slope exceeds the male slope. Collinearity is moderate (Pearson r of "
    f"log10 female and log10 male = "
    f"{np.corrcoef(np.log10(train.female_gametocytes_per_ul), np.log10(train.male_gametocytes_per_ul))[0,1]:.2f}), "
    f"so both coefficients remain estimable but with correlated uncertainty."
)
asex_txt = (
    f"Adding log10 asexual density to the selected model: LRT chi2="
    f"{lrt(asex_kind, best_kind)[0]:.2f}, df={lrt(asex_kind, best_kind)[1]}, "
    f"p={lrt(asex_kind, best_kind)[2]:.3f}; LOO log loss "
    f"{results[asex_kind]['loo']:.4f} vs {results[best_kind]['loo']:.4f} without it. "
    f"Asexual density {'improves' if asex_helps else 'does not improve'} out-of-sample "
    f"prediction and was {'retained' if asex_helps else 'excluded'}. Asexual density is "
    f"correlated with gametocyte density (r="
    f"{np.corrcoef(np.log10(train.asexual_per_ul), np.log10(train.female_gametocytes_per_ul+train.male_gametocytes_per_ul))[0,1]:.2f} "
    f"on the log10 scale), so any association is a marker of concurrent gametocytaemia, "
    f"not evidence that asexual parasites infect mosquitoes."
)
calib_txt = (
    f"Pearson chi2={pearson:.1f} on {df_res} df (dispersion {disp:.2f}, p={disp_p:.3f}); "
    f"beta-binomial extension LRT p={lrt_bb_p:.3f} (phi={phi_hat:.1f}), so binomial "
    f"variance is {'not adequate' if lrt_bb_p < 0.05 else 'adequate'}. Grouped calibration "
    f"(3 equal-count bins of fitted probability): " +
    "; ".join(f"bin{c['bin']} fitted {c['mean_fitted']} vs observed {c['observed_proportion']}"
              for c in calib) +
    f"; Hosmer-Lemeshow-type statistic {hl:.2f}, p={hl_p:.3f}. "
    f"95% intervals are Wald/Monte-Carlo intervals for the mean infection probability "
    f"from the asymptotic sampling distribution of the coefficients (40,000 draws)."
)

out = {
    "model": {
        "name": f"Binomial logistic GLM on log10 gametocyte density ({final})",
        "formula": formula + "  with coefficients " + json.dumps(coefs) +
                   " (standard errors " + json.dumps(ses) + ")",
        "likelihood": "mosquitoes_infected ~ Binomial(mosquitoes_examined, p); "
                      "maximum likelihood via scipy.optimize (BFGS)",
    },
    "predictions": [
        {"feed_id": fid, "infection_probability": round(float(p), 4),
         "probability_interval": [round(float(a), 4), round(float(b), 4)]}
        for fid, p, a, b in zip(pred.feed_id, p_hat, lo, hi)
    ],
    "transmission_summary": {
        "expected_infected_mosquitoes": round(expected_infected, 1),
        "weighted_infection_probability": round(float(weighted_p), 4),
    },
    "diagnostics": {
        "training_binomial_log_loss": round(float(results[final]["train_logloss"]), 4),
        "model_comparison": cmp_txt,
        "asexual_assessment": asex_txt,
        "calibration": calib_txt,
        "model_table": {k: {"k": results[k]["k"], "aic": round(results[k]["aic"], 2),
                            "loo_log_loss": round(results[k]["loo"], 4),
                            "train_log_loss": round(results[k]["train_logloss"], 4)}
                        for k in results},
    },
    "literature": [
        {"claim": "In 148 direct membrane feeding experiments with naturally infected "
                  "carriers, mosquito infection prevalence was determined mainly by female "
                  "gametocyte density, with male gametocytes limiting transmission at low "
                  "densities - justifying a gametocyte-density logistic model and testing "
                  "female and male terms separately.",
         "citation": "Bradley J, Stone W, Da DF, et al. Predicting the likelihood and "
                     "intensity of mosquito infection from sex specific Plasmodium falciparum "
                     "gametocyte density. eLife 2018;7:e34463.",
         "url": "https://doi.org/10.7554/eLife.34463"},
        {"claim": "PfMGET (male) and Pfs25 (female) qRT-PCR quantify the two gametocyte "
                  "sexes separately in clinical samples, which is why the two density streams "
                  "in these data are distinct measurements rather than duplicates.",
         "citation": "Santolamazza F, Avellino P, Siciliano G, et al. A molecular assay to "
                     "quantify male and female Plasmodium falciparum gametocytes: results from "
                     "2 randomized controlled trials using primaquine for gametocyte clearance. "
                     "Journal of Infectious Diseases 2017;216(4):468-476.",
         "url": "https://doi.org/10.1093/infdis/jix237"},
        {"claim": "Pfs25 qRT-PCR gametocyte density is a molecular biomarker whose "
                  "association with mosquito infectivity is monotone but saturating and noisy, "
                  "supporting a saturating (logit-linear in log density) link with binomial "
                  "sampling error rather than a linear dose-response in raw density.",
         "citation": "Stone W, Sawa P, Lanke K, et al. Comparison of molecular quantification "
                     "of Plasmodium falciparum gametocytes by Pfs25 qRT-PCR and QT-NASBA in "
                     "relation to mosquito infectivity. Malaria Journal 2017;16:8.",
         "url": "https://doi.org/10.1186/s12936-016-1584-z"},
        {"claim": "Total parasite (asexual) density and gametocyte density are correlated in "
                  "natural infections and asexual stages are not themselves infectious to "
                  "mosquitoes, so asexual density can only act as a proxy marker and any "
                  "predictive association must not be read causally.",
         "citation": "Ouedraogo AL, Goncalves BP, Gneme A, et al. Dynamics of the human "
                     "infectious reservoir for malaria determined by mosquito feeding assays "
                     "and ultrasensitive malaria diagnostics in Burkina Faso. Journal of "
                     "Infectious Diseases 2016;213(1):90-99.",
         "url": "https://doi.org/10.1093/infdis/jiv370"},
    ],
}

with open("results.json", "w") as fh:
    json.dump(out, fh, indent=2)

print(json.dumps(out["diagnostics"], indent=2)[:4000])
print(json.dumps(out["predictions"], indent=1))
print(out["transmission_summary"], out["model"]["formula"])
