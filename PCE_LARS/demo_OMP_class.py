"""Implements the selection method of a polynomial chaos expansion algorithm in Python.

This implements 2 algorithms:
- Algorithm B.1 page 628 of (Lüthen, et al., 2021),
- Algorithm B.1 with with CV using Corrected Leave-One-Out or K-Fold.

TODO-List
---------
- Extend the code to multiple output dimensions.
- Early stopping of the algorithm using the K-Fold or Corrected LOO score.
- Implement LARS using algorithm B.2.

Reference
---------
- Lüthen, N., Marelli, S., & Sudret, B. (2021).
  Sparse polynomial chaos expansions: Literature survey and benchmark.
  SIAM/ASA Journal on Uncertainty Quantification, 9(2), 593-649.
- https://gist.github.com/mbaudin47/87a09578aef2e38b498f2f5c5cda193b

Output
------
Fitting score = 1.0005e+00
Current active indices = [0]
  Best index = 30 with max. abs. corr. = 2.2630e+00
  Fitting score = 7.3706e-01
Current active indices = [0, 30]
  Best index = 1 with max. abs. corr. = 1.7322e+00
  Fitting score = 5.3630e-01
Current active indices = [0, 30, 1]
  Best index = 15 with max. abs. corr. = 1.5058e+00
  Fitting score = 3.9255e-01
Current active indices = [0, 30, 1, 15]
  Best index = 77 with max. abs. corr. = 1.5628e+00
  Fitting score = 2.6353e-01
Current active indices = [0, 30, 1, 15, 77]
  Best index = 10 with max. abs. corr. = 1.2747e+00
  Fitting score = 1.4707e-01
Current active indices = [0, 30, 1, 15, 77, 10]
  Best index = 40 with max. abs. corr. = 1.0375e+00
  Fitting score = 6.7582e-02
Current active indices = [0, 30, 1, 15, 77, 10, 40]
  Best index = 7 with max. abs. corr. = 6.7762e-01
  Fitting score = 3.7777e-02

[...]

Current active indices = [0, 30, 1, 15, 77, 10, 40, 7, 98, 49, 35, 89, 18, 32, 46, 63, 26, 29, 91, 19, 41, 96, 81, 47, 72, 82, 85, 94, 8, 3, 71, 22, 74, 28, 34, 2, 48, 66, 5, 51, 17, 64, 9, 57, 97, 69, 39, 12, 24, 20, 52, 73, 58, 50, 79, 60, 25, 87, 75, 95, 45, 70, 31, 44, 61, 38, 62, 43, 86, 54, 65, 99, 33, 37, 59, 6, 16, 80, 55, 42, 78, 56, 11, 36, 76, 67, 92, 14, 93, 4, 13, 84, 83, 21, 23, 53, 90, 27]
  Best index = 88 with max. abs. corr. = 3.0559e-03
  Fitting score = 1.4961e-01
Current active indices = [0, 30, 1, 15, 77, 10, 40, 7, 98, 49, 35, 89, 18, 32, 46, 63, 26, 29, 91, 19, 41, 96, 81, 47, 72, 82, 85, 94, 8, 3, 71, 22, 74, 28, 34, 2, 48, 66, 5, 51, 17, 64, 9, 57, 97, 69, 39, 12, 24, 20, 52, 73, 58, 50, 79, 60, 25, 87, 75, 95, 45, 70, 31, 44, 61, 38, 62, 43, 86, 54, 65, 99, 33, 37, 59, 6, 16, 80, 55, 42, 78, 56, 11, 36, 76, 67, 92, 14, 93, 4, 13, 84, 83, 21, 23, 53, 90, 27, 88]
  Best index = 68 with max. abs. corr. = 1.2446e-03
  Fitting score = 1.5592e-01
"""

# %%
import openturns as ot
import OrthogonalMatchingPursuitPCE as omppce
from openturns.usecases import ishigami_function
import openturns.viewer as otv
import numpy as np

# %%
ot.RandomGenerator.SetSeed(0)

# %%
im = ishigami_function.IshigamiModel()
sample_size = 200
experiment = ot.MonteCarloExperiment(im.inputDistribution, sample_size)
input_sample, wX = experiment.generateWithWeights()
output_sample = im.model(input_sample)

# %%
# Create basis
input_dimension = im.inputDistribution.getDimension()
basis = ot.OrthogonalProductPolynomialFactory(
    [im.inputDistribution.getMarginal(i) for i in range(input_dimension)]
)

# %%
maximumBasisSize = 100
print(f"Number of coefficients = {maximumBasisSize}")

# %%
# [Markdown]
# KFold : Cholesky
# CorrectedLeaveOneOut : Nécessite la diagonal de l'inverse de la
# matrice H > Choisir SVD

# %%
# Set minAbsCorrelation to zero to see all path.
algo = omppce.OrthogonalMatchingPursuitPCE(
    input_sample,
    output_sample,
    im.inputDistribution,
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
print("Selection History:", algo.getSelectionHistory())
fitting_score_history = algo.getFittingScoreHistory()
print("Fitting Score History:", fitting_score_history)
result = algo.getResult()
result

# %%
fitting = algo.getFittingAlgorithm()

# %%
input_test = im.inputDistribution.getSample(1000)
output_test = im.model(input_test)
meta_model = result.getMetaModel()
validation = ot.MetaModelValidation(output_test, meta_model(input_test))
print(f"Q2 = {validation.computeR2Score()[0]:.15f}")


# %%
error_factor = ot.ResourceMap.GetAsScalar("SparseMethod-MaximumErrorFactor")
marginal_fitting_score = fitting_score_history.getMarginal(0).asPoint()
min_index = np.argmin(marginal_fitting_score)
fitting_score_min = min(marginal_fitting_score)

# %%
# Plot the KFold score as a function of the iterations
graph = ot.Graph(
    f"OMP with {fitting.getClassName()}",
    "Iteration",
    f"{fitting.getClassName()} score",
    True,
)
number_of_selected_coefficients = len(fitting_score_history)
cloud = ot.Cloud(range(number_of_selected_coefficients), marginal_fitting_score)
graph.add(cloud)
graph.setLogScale(ot.GraphImplementation.LOGY)
# Plot min corrected score
cloud = ot.Cloud([min_index], [fitting_score_min])
cloud.setPointStyle("circle")
cloud.setLegend("Min")
graph.add(cloud)
# Plot error factor
curve = ot.Curve(
    [0, number_of_selected_coefficients], [error_factor * fitting_score_min] * 2
)
curve.setLineWidth(2.0)
curve.setLegend("Treshold")
graph.add(curve)
view = otv.View(graph)
view.save("demo_OMP_class.png")

# %%
