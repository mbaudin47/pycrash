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
from openturns.usecases import ishigami_function
import openturns.viewer as otv

# %%
# [Markdown]
# Notice that this script does not use Numpy _at all_.

# %%
ot.RandomGenerator.SetSeed(0)

# %%
im = ishigami_function.IshigamiModel()
sample_size = 200
input_sample = im.inputDistribution.getSample(sample_size)
output_sample = im.model(input_sample)

# %%
maximum_basis_dimension = 100
leastSquaresMethodName = "SVD"

# %%
# Create basis
basis = ot.OrthogonalProductPolynomialFactory(
    [
        im.inputDistribution.getMarginal(i)
        for i in range(im.inputDistribution.getDimension())
    ]
)
basis

# %%
# 1. Compute all coefficients
transformation = ot.DistributionTransformation(im.inputDistribution, basis.getMeasure())
standard_input = transformation(input_sample)
indices = ot.Indices(maximum_basis_dimension)
indices.fill()
functions = [basis.build(i) for i in indices]
designProxy = ot.DesignProxy(standard_input, functions)
leastSquaresSolver = ot.LeastSquaresMethod.Build(
    leastSquaresMethodName, designProxy, range(len(indices))
)
outputDimension = output_sample.getDimension()
coefficients = ot.Sample(len(indices), outputDimension)
for j in range(outputDimension):
    coeffsJ = leastSquaresSolver.solve(output_sample.getMarginal(j).asPoint())
    for i in range(len(indices)):
        coefficients[i, j] = coeffsJ[i]
# Create the result
result = ot.FunctionalChaosResult(
    input_sample,
    output_sample,
    im.inputDistribution,
    transformation,
    transformation.inverse(),
    basis,
    indices,
    coefficients,
    functions,
)
result

# %%
input_test = im.inputDistribution.getSample(1000)
output_test = im.model(input_test)
meta_model = result.getMetaModel()
validation = ot.MetaModelValidation(output_test, meta_model(input_test))
print(f"Q2 = {validation.computeR2Score()[0]:.15f}")

# %%
# 2. Compute the coefficients using Orthogonal Matching Pursuit (OMP) method
sample_size = standard_input.getSize()
transformation = ot.DistributionTransformation(im.inputDistribution, basis.getMeasure())
standard_input = transformation(input_sample)
# Create a list of functions
functions = [functions[i] for i in range(maximum_basis_dimension)]
designProxy = ot.DesignProxy(standard_input, functions)
# Initialisation
list_of_active_functions = [0]  # Initialize with constant basis
leastSquaresMethod = ot.LeastSquaresMethod.Build(
    leastSquaresMethodName, designProxy, list_of_active_functions
)
residuals = output_sample.asPoint()
# Select your best fitting algorithm
# fitting = ot.CorrectedLeaveOneOut()
kParameter = 10
fitting = ot.KFold(kParameter)
# Compute initial fitting score
fitting_score = fitting.run(
    standard_input,
    output_sample,
    ot.Point(sample_size, 1) / sample_size,
    functions,
    list_of_active_functions,
)
print(f"  Fitting score = {fitting_score:.4e}")
fitting_score_list = [fitting_score]
# Update residuals
residuals -= ot.Point(sample_size, output_sample.computeMean()[0])
# TODO: Repeat this for output each marginal
for i in range(maximum_basis_dimension - 1):
    # Find candidate with maximum absolute correlation with the residual
    print(f"Current active indices = {list_of_active_functions}")
    maximum_absolute_correlation = 0.0
    best_basis_function_index = None
    for j in range(maximum_basis_dimension):
        if j in list_of_active_functions:
            # Skip this basis (already active)
            continue
        current_basis_function = functions[j]
        basis_function_value = current_basis_function(standard_input)
        current_absolute_correlation = (
            abs(residuals.dot(basis_function_value.asPoint())) / sample_size
        )
        if current_absolute_correlation > maximum_absolute_correlation:
            best_basis_function_index = j
            maximum_absolute_correlation = current_absolute_correlation
    print(
        f"  Best index = {best_basis_function_index} "
        f"with max. abs. corr. = {maximum_absolute_correlation:.4e}"
    )
    # Update the LS method
    leastSquaresMethod.update([best_basis_function_index], list_of_active_functions, [])
    # Add the best candidate to the active set
    list_of_active_functions.append(best_basis_function_index)
    # Update the coefficients
    coefficients = leastSquaresMethod.solve(output_sample.asPoint())
    # Update the residuals
    designMatrix = leastSquaresMethod.computeWeightedDesign()
    residuals = output_sample.asPoint() - designMatrix * coefficients
    # Compute corrected leave-out score
    # After https://github.com/openturns/openturns/issues/2948
    fitting_score = fitting.run(leastSquaresMethod, output_sample)
    print(f"  Fitting score = {fitting_score:.4e}")
    fitting_score_list.append(fitting_score)

coefficientSample = ot.Sample.BuildFromPoint(coefficients)

# Réserve
# Create the result
functions = [functions[i] for i in list_of_active_functions]
result = ot.FunctionalChaosResult(
    input_sample,
    output_sample,
    im.inputDistribution,
    transformation,
    transformation.inverse(),
    basis,
    list_of_active_functions,
    coefficientSample,
    functions,
)
result

# %%
meta_model = result.getMetaModel()
validation = ot.MetaModelValidation(output_test, meta_model(input_test))
print(f"Q2 = {validation.computeR2Score()[0]:.15f}")

# %%
def argmin(liste):
    # This can be avoided if using np.argmin.
    # But we want to show that Numpy can be avoided here,
    # and rely only on OpenTURNS for the OMP algorithm.
    if not liste:
        return None
    
    indice_min = 0
    valeur_min = liste[0]
    
    for i in range(1, len(liste)):
        if liste[i] < valeur_min:
            valeur_min = liste[i]
            indice_min = i
            
    return indice_min

# %%
threshold = ot.ResourceMap.GetAsScalar("SparseMethod-ErrorThreshold")
error_factor = ot.ResourceMap.GetAsScalar("SparseMethod-MaximumErrorFactor")
min_index = argmin(fitting_score_list)
fitting_score_min = min(fitting_score_list)
graph = ot.Graph(
    f"{fitting.getClassName()}", "Iteration", f"{fitting.getClassName()} score", True
)
cloud = ot.Cloud(range(maximum_basis_dimension), fitting_score_list)
graph.add(cloud)
graph.setLogScale(ot.GraphImplementation.LOGY)
# Plot min corrected score
cloud = ot.Cloud([min_index], [fitting_score_min])
cloud.setPointStyle("circle")
cloud.setLegend("Min")
graph.add(cloud)
# Plot error factor
curve = ot.Curve([0, maximum_basis_dimension], [error_factor * fitting_score_min] * 2)
curve.setLineWidth(2.0)
curve.setLegend("Treshold")
graph.add(curve)
view = otv.View(graph)
view.save("test_orthogonal_matching_pursuit.png")

# %%
