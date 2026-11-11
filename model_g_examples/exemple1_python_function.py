import openturns as ot


def my_model(X):
    """Compute the deflection of a cantilever beam subjected to a point load.

    Parameters
    ----------
    X : sequence of float of length 4
        Input parameters containing:
        - L : float, length of the beam.
        - F : float, point load applied at the end.
        - E : float, Young's modulus of the material.
        - I : float, second moment of area of the cross-section.

    Returns
    -------
    f : list of float of length 1
        Deflection at the end of the beam.
    """
    L, F, E, I = X
    f = (F * L**3) / (3 * E * I)
    return [f]


# 1. Define the physical model
dimension = 4
input_function = ot.PythonFunction(dimension, 1, my_model)

# 2. Define the input distributions
distribution_L = ot.Normal(2.5, 0.05)
distribution_F = ot.Normal(3000.0, 300.0)
distribution_E = ot.Normal(2.1e11, 1e10)
distribution_I = ot.Normal(4.0e-6, 1e-7)

input_distribution = ot.JointDistribution(
    [distribution_L, distribution_F, distribution_E, distribution_I]
)

# 3. Propagate uncertainties
input_design = ot.MonteCarloExperiment(input_distribution, 1000)
input_sample = input_design.generate()
output_sample = input_function(input_sample)

if __name__ == "__main__":
    print("Output sample (first 5 rows):")
    print(output_sample[:5])