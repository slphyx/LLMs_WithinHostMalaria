#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import os

# --- Inputs ---
asexual_path = 'input/asexual_driver.csv'
gametocyte_path = 'input/gametocytes.csv'
prediction_path = 'input/prediction_times.csv'

# Read inputs
asexual_df = pd.read_csv(asexual_path)
gametocyte_df = pd.read_csv(gametocyte_path)
prediction_times = pd.read_csv(prediction_path)['time_days'].values

# --- Interpolation ---
asexual_times = asexual_df['time_days'].values
asexual_values = asexual_df['asexual_parasites_per_ml'].values

def interpolate_asexual(t, asexual_times, asexual_values):
    return np.interp(t, asexual_times, asexual_values)

# --- Model Equations ---
def model(t, state, asexual_times, asexual_values, k, r_early, r_late, lambda_):
    A = interpolate_asexual(t, asexual_times, asexual_values)
    I_early, I_late, M = state
    
    dI_early_dt = k * A - r_early * I_early
    dI_late_dt = r_early * I_early - r_late * I_late
    dM_dt = r_late * I_late - lambda_ * M
    
    return [dI_early_dt, dI_late_dt, dM_dt]

# --- Solver Setup ---
start_day = 0
end_day = 26
rtol = 1e-6
atol = 1e-8

# Placeholder parameters (to be estimated later)
k = 0.01
r_early = 0.1
r_late = 0.05
lambda_ = 0.05

# Initial conditions
initial_state = [0, 0, 0]

# Solve ODE
sol = solve_ivp(model, [start_day, end_day], initial_state,
                args=(asexual_times, asexual_values, k, r_early, r_late, lambda_),
                rtol=rtol, atol=atol, method='RK45')

# --- Generate Trajectory ---
trajectory_df = pd.DataFrame({
    'time_days': sol.t,
    'mature_gametocytes_per_ml': sol.y[2]
})
trajectory_df.to_csv('trajectory.csv', index=False)

# --- Placeholder for Parameter Estimation ---
# In practice, estimate parameters using gametocyte data and held-out predictions.
print("Trajectory generated successfully.")