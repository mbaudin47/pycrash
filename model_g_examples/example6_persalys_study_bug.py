#!/usr/bin/env python
"""
Persalys script setting up a logistic growth physical model and a Monte Carlo central tendency analysis.
"""

import openturns as ot
import persalys

example6_persalys_study_bug = persalys.Study('example6_persalys_study_bug')
dist_y0 = ot.Normal(3.9, 0.39)
y0 = persalys.Input('y0', 3.9, dist_y0, 'Initial population (in millions)')
dist_a = ot.Normal(0.03134, 0.003134)
a = persalys.Input('a', 0.03134, dist_a, 'Growth rate (1/year)')
dist_b = ot.Normal(1.5887e-10, 1.5887e-11)
b = persalys.Input('b', 1.5887e-10, dist_b, 'Saturation coefficient (1/million)')
t0 = persalys.Input('t0', 1790, 'Initial year')
t_final = persalys.Input('t_final', 2000, 'Final year')
final_population = persalys.Output('final_population', '')
inputs = [y0, a, b, t0, t_final]
outputs = [final_population]
code = r'''

import numpy as np
from scipy.integrate import solve_ivp

# Implementation of the physical model for Persalys (Python Interface)

def logistic_rhs(t, y, a, b):
    """Right-hand side of the logistic ordinary differential equation.

    Parameters
    ----------
    t : float
        Time step (year).
    y : ndarray
        Current population state vector.
    a : float
        Growth rate (1/year).
    b : float
        Saturation coefficient (1/million).

    Returns
    -------
    dydt : float
        Derivative of the population with respect to time.
    """
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
    final_population : float
        Final population evaluated at the end of the time period (in millions).
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
PythonModel_0 = persalys.PythonPhysicalModel('PythonModel_0', inputs, outputs, code)
example6_persalys_study_bug.add(PythonModel_0)
centralTendency_0 = persalys.MonteCarloAnalysis('centralTendency_0', PythonModel_0)
centralTendency_0.setMaximumCoefficientOfVariation(0.01)
centralTendency_0.setMaximumElapsedTime(60)
centralTendency_0.setBlockSize(8)
centralTendency_0.setSeed(0)
centralTendency_0.setAllowFailedEvaluations(False)
example6_persalys_study_bug.add(centralTendency_0)
persalys.Study.Add(example6_persalys_study_bug)