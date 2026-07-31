"""
Check the OrthogonalMatchingPursuitPCE class.

This script is based on a function which has an exact decomposition
onto Legendre polynomials.
In this case, the OMP algorithm must find the functions in the expected
order with the expected coefficients.
The test function has a 1D input dimension.
This does not prevent it from accurately testing the algorithm,
which is mainly insensitive to the input dimension of the function,
from the algorithmic point of view.
Notice, however, that the accuracy of the algorithm depends on the dimension
(but not the structure of the loops).
"""

# %%
import openturns as ot
import OrthogonalMatchingPursuitPCE as omppce
import numpy.testing as npt
import numpy as np


# %%
class LegendreFunction(ot.OpenTURNSPythonFunction):
    def __init__(self, degree):
        # Workaround for https://github.com/openturns/openturns/issues/2671
        # A ProductPolynomialEvaluation is not converted into a Function
        super().__init__(1, 1)
        self.setInputDescription(["X"])
        self.setOutputDescription(["Y"])
        self.degree = degree
        self.polynomial = ot.LegendreFactory().build(degree)

    def _exec(self, X):
        Y = self.polynomial(X[0])
        return [Y]


# %%
# 1. Définition de la loi d'entrée (Uniforme sur [-1, 1] pour Legendre)
distribution = ot.Uniform(-1.0, 1.0)

# %%
# 2. Création de la base de polynômes de Legendre
# La famille d'orthonormalisation associée à la loi Uniforme est Legendre
poly_factory = ot.LegendreFactory()

# %%
# 3. Définition des indices des fonctions et des coefficients
# Nous choisissons des coefficients avec des valeurs absolues distinctes
# pour anticiper l'ordre de sélection du LARS (du plus grand au plus petit)
indices = [1, 3, 4]
coefficients = [5.0, 10.0, 1.0]

# %%
# On construit la liste des fonctions correspondantes : 10*P3 + 5*P1 + 1*P4
functions_list = [ot.Function(LegendreFunction(i)) for i in indices]

# %%
# 4. Création de la fonction via LinearCombinationFunction
# Cette classe crée une fonction f(x) = sum( coeff_i * phi_i(x) )
test_function = ot.LinearCombinationFunction(functions_list, coefficients)

# %%
# --- Vérification du fonctionnement ---
sample_size = 100
experiment = ot.MonteCarloExperiment(distribution, sample_size)
input_sample, wX = experiment.generateWithWeights()
output_sample = test_function(input_sample)

# %%
print("Entrées (X) :\n", input_sample[:5])
print("Sorties calculées (Y) :\n", output_sample[:5])

# %%
basis = ot.OrthogonalProductPolynomialFactory([distribution])

# %%
# Set minAbsCorrelation to zero to see all path.
maximumBasisSize = 10
algo = omppce.OrthogonalMatchingPursuitPCE(
    input_sample,
    output_sample,
    distribution,
    basis,
    maximumBasisSize,
    wX=wX,
    verbose=True,
    leastSquaresMethodName="Cholesky",
    minAbsCorrelation=0.0,  # Arbitrary early stopping
)
algo.run()

# %%
# Display outputs
print("Active Indices:", algo.getActiveIndices())
selection_history = algo.getSelectionHistory()
print("Selection History:", selection_history)
fitting_score_list = algo.getFittingScoreHistory()
print("Fitting Score History:", fitting_score_list)
result = algo.getResult()
result

# %%
# Compute non zero coefficients
absolute_tolerance = 1.0e-12
computed_indices = result.getIndices()
computed_coefficients = result.getCoefficients()
nonzero_computed_indices = []
nonzero_computed_coefficients = []
number_of_nonzero_coefficients = 0
for i in range(len(computed_indices)):
    if abs(computed_coefficients[i, 0]) > absolute_tolerance:
        nonzero_computed_indices.append(computed_indices[i])
        nonzero_computed_coefficients.append(computed_coefficients[i, 0])
        number_of_nonzero_coefficients += 1

# %%
# Check indices and coefficients
assert indices == nonzero_computed_indices
npt.assert_almost_equal(coefficients, nonzero_computed_coefficients)

# %%
# Check selection history
# 1. Compute the expected indices (by sorting the coefficients in decreasing order)
sorting_indices = np.argsort(-np.array(coefficients))
sorting_indices = [int(i) for i in sorting_indices]
expected_active_history = [0] + [indices[i] for i in sorting_indices]
# 2. Get the computed marginal selection history
marginal_selection_history = selection_history[0]
# 3. Extract the indices corresponding to the first significant coefficients
computed_marginal_selection_history = marginal_selection_history[0:number_of_nonzero_coefficients + 1]
assert computed_marginal_selection_history == expected_active_history

# %%

# %%
