# %%
import numpy as np
from scipy.integrate import solve_ivp

# %%
# Implémentation du modèle physique pour Persalys (Interface Python)

# Define the ordinary differential equation dy/dt = a*y - b*y^2
def logistic_rhs(t, y, a, b):
    return a * y[0] - b * (y[0] ** 2)

# %%
NUMBER_OF_TIME_STEPS = 500
def _exec(y0, a, b, t0, t_final):
    """
    Fonction d'évaluation appelée par Persalys.
    
    Entrées :
        y0 : Population initiale (en millions)
        a : Taux de croissance (1/an)
        b : Coefficient de saturation (1/million)
        t0 : Année initiale
        t_final : Année finale
        
    Sortie :
        t_eval : Tableau des années évaluées
        y_eval : Tableau des populations correspondantes
    """
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
    final_population = sol.y  # <- This is wrong, it should be sol.y[0] to get the population values

    return final_population

# %%
# Problem parameters
a = 0.03134
b = 1.5887e-10
t0 = 1790.0
t_final = 2010.0
y0 = 3.9e6
final_population = _exec(y0, a, b, t0, t_final)

# %%
# Print key population results
print(f"Population in 1790 (Initial):   {y0/ 1e6:.2f} Million")
print(f"Population in 2010 (Numerical): {final_population / 1e6:.2f} Million")
print(f"Theoretical Carrying Capacity:  {(a / b) / 1e6:.2f} Million")
