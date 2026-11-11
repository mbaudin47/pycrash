#!/usr/bin/env python

import openturns as ot
import persalys

exemple5_persalys_study = persalys.Study('exemple5_persalys_study')
dist_y0 = ot.Normal(3.9, 0.39)
y0 = persalys.Input('y0', 3.9, dist_y0, 'Initial population (in millions)')
dist_a = ot.Normal(0.03134, 0.003134)
a = persalys.Input('a', 0.03134, dist_a, 'Growth rate (1/year)')
dist_b = ot.Normal(1.5887e-10, 1.5887e-11)
b = persalys.Input('b', 1.5887e-10, dist_b, 'Saturation coefficient (1/million)')
t0 = persalys.Input('t0', 1790, 'Initial year')
t_final = persalys.Input('t_final', 2000, 'Final year')
final_population = persalys.Output('final_population', '')
final_population.setValue(184.636634495)
inputs = [y0, a, b, t0, t_final]
outputs = [final_population]
code = r'''
import numpy as np
from scipy.integrate import solve_ivp

# Implémentation du modèle physique pour Persalys (Interface Python)

# Define the ordinary differential equation dy/dt = a*y - b*y^2
def logistic_rhs(t, y, a, b):
    return a * y[0] - b * (y[0] ** 2)

NUMBER_OF_TIME_STEPS = 500
def _exec(y0, a, b, t0, t_final):
    """Evaluation function called by Persalys.
    
    Parameters
    ----------
    y0 : float
        Initial population (in millions).
    a : float
        Growth rate (1/year).
    b : float
        Saturation coefficient (1/million).
    t0 : float
        Initial year.
    t_final : float
        Final year.
    
    Returns
    -------
    y_eval : ndarray
        Array of corresponding populations.
    """
    # Scale y0 from millions into actual population
    y0 *= 1.e6
    
    # Time grid for output evaluation
    t_eval = np.linspace(t0, t_final, NUMBER_OF_TIME_STEPS)

    # Solve ODE using SciPy's explicit Runge-Kutta solver (RK45)
    sol = solve_ivp(
        fun=logistic_rhs,
        t_span=(t0, t_final),
        y0=[y0],
        args=(a, b),
        t_eval=t_eval,
        method='RK45'
    )
    final_population = sol.y[0][-1]
    
    # Scale back into millions
    final_population /= 1.e6
    return final_population
'''
ModèlePython_0 = persalys.PythonPhysicalModel('ModèlePython_0', inputs, outputs, code)
exemple5_persalys_study.add(ModèlePython_0)
tendanceCentrale_0 = persalys.MonteCarloAnalysis('tendanceCentrale_0', ModèlePython_0)
tendanceCentrale_0.setMaximumCoefficientOfVariation(0.01)
tendanceCentrale_0.setMaximumElapsedTime(60)
tendanceCentrale_0.setBlockSize(8)
tendanceCentrale_0.setSeed(0)
tendanceCentrale_0.setAllowFailedEvaluations(False)
exemple5_persalys_study.add(tendanceCentrale_0)
persalys.Study.Add(exemple5_persalys_study)
