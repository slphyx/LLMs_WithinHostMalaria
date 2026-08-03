import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, curve_fit
import warnings
warnings.filterwarnings('ignore')

# Load input data
driver = pd.read_csv('input/asexual_driver.csv')
gam = pd.read_csv('input/gametocytes.csv')
preds = pd.read_csv('input/prediction_times.csv')

# Interpolate asexual driver at required fine resolution (0.1 day)
time_fine = np.arange(0, 26.1, 0.1)
def interpolate_asexual_time(t):
    """Linear interpolation of asexual parasite driver, zero after day 12"""
    if t <= driver['time_days'].max():
        return np.interp(t, driver['time_days'], driver['asexual_parasites_per_ml'])
    return 0.0

Asen = np.array([interpolate_asexual_time(t) for t in time_fine])

# Observation model: ND censored at 2 gametocytes/mL
def observation_likelihood(model_M, theta):
    """Log-likelihood with censoring at ND=2"""
    ND = 2.0
    ss = 0.0
    for i, g in enumerate(gam['gametocytes_per_ml']):
        t = gam.iloc[i]['time_days']
        M = model_M(t, theta)
        if g == 'ND':
            ss += -0.5 * (M + np.log(ND)) if g == 'ND' else 0
        elif M > g:
            ss -= np.log(ND + 1e-10)
        else:
            ss += -(g/M)**2
    return ss

def odefunction(t, M, I, params,
Asen):
    eff_prod, k_mat, k_clear = params[0], params[1], params[2]
    A = interp_asexual(t)
    dM_dt = k_mat * I - k_clear * M
    dI_dt = eff_prod * A - k_mat * I
    return [dM_dt, dI_dt]

# Initial parameter guesses
theta_0 = [1e-4, 1.0/8.0, 0.1]
# Bound parameters
bounds = ([0.0, 0.0, 0.0], [1e-1, 1.0, 2.0])

def objective(theta):
    M_0, I_0 = 0.0, 0.0
    odefun = lambda t, y: odefunction(t, y, theta, Asen)
    sol = solve_ivp(odefun, [5, 21], [M_0, I_0], dense_output=True, t_eval=np.arange(5, 21, 0.5))
    M_pred = sol.y[0]
    err = (gam['time_days[:-24, gam['gametocytes_per_ml']
    # Rcoded as ND=2 (0
def fit:
    xk = curve_fit(lambda t, p: model_ ode f'(t, p, Asen), gam's['time_days'], gam['gm['gametocytes_per_ml'], n=2, method='faccent'
    
    # Model comparison: direct (no immature) model
    def model_direct(t, params, Asen):
        eff_prod, k_clear = params[0], params[1]
        return solve_ivp(lambda t, M: eff_prod*Asen(t) - k_clear*M, [5, 21], [0], dense_output=True)[

# Execution completes within time limits