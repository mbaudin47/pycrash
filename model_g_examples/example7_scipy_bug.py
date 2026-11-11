# %%
import numpy as np
import openturns as ot
from scipy.integrate import solve_ivp

# %%
NUMBER_OF_TIME_STEPS = 500


# Define the ordinary differential equation dy/dt = a*y - b*y^2
def logistic_rhs(t, y, a, b):
    return a * y[0] - b * (y[0] ** 2)


# %%
# 1. Definition of the evaluation function
def _exec(x):
    """Returns the final population at final time.

    Parameters
    ----------
    x : list
        A list containing [y0, a, b, t0, t_final].

    Returns
    -------
    y : list
        A list of 1 item, containing the population at final time.
    """
    y0, a, b, t0, t_final = x
    # Scale y0 from millions into actual population
    y0 *= 1.0e6

    # Time grid for output evaluation
    t_eval = np.linspace(t0, t_final, NUMBER_OF_TIME_STEPS)

    # Solve ODE using SciPy's explicit Runge-Kutta solver (RK45)
    sol = solve_ivp(
        fun=logistic_rhs,
        t_span=(t0, t_final),
        y0=[y0],
        args=(a, b),
        t_eval=t_eval,
        method="RK45",
    )
    final_population = (
        sol.y
    )  # <- This is wrong, it should be sol.y[0][-1] to get the population values
    # Correct:
    # final_population = sol.y[0][-1]

    # Scale back into millions
    final_population /= 1.0e6
    print(f"Type: {type(final_population)}")
    # Correct:
    # return [final_population]
    return final_population # <- Wrong


# %%
dist_y0 = ot.Normal(3.9, 0.39)
dist_a = ot.Normal(0.03134, 0.003134)
dist_b = ot.Normal(1.5887e-10, 1.5887e-11)


# %%
# 2. Function instantiation and encapsulation
g_model_py = ot.PythonFunction(5, 1, _exec)
g_model_py.setInputDescription(["y0", "a", "b", "t0", "t_final"])
g_model_py.setOutputDescription(["final_population"])
g_model = ot.Function(g_model_py)

# %%
t0 = 1790.0
t_final = 2000.0
indices = [3, 4]
reference_value = [t0, t_final]
g_model_parametric = ot.ParametricFunction(g_model, indices, reference_value)

# %%
# 3. Definition of the joint input distribution
dimension = g_model.getInputDimension()
distribution = ot.JointDistribution([dist_y0, dist_a, dist_b])

# %%
# 4. Evaluation on a design of experiments
sample_input = distribution.getSample(100)
sample_output = g_model_parametric(sample_input)

# %%
print("Output sample (first 5 rows):")
print(sample_output[:5])

# %%
