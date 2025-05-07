"""Implement the LARS algorithm in Python"""

# %%
import openturns as ot
from openturns.usecases import ishigami_function

# %%
im = ishigami_function.IshigamiModel()
sample_size = 20
input_sample = im.inputDistribution.getSample(sample_size)
output_sample = im.model(input_sample)

# %%
maximum_basis_dimension = 10
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
    for i in range(len(indices)):  # coefficients[:,j] = coeffsJ
        coefficients[i, j] = coeffsJ[i]
# Create the result
# The physical model is unknown in this case ...
physicalModel = ot.Function()
# ... which implies that the composed model is unknown in this case
composedModel = ot.Function()
residualsPoint = [-1.0]
relativeErrorsPoint = [-1.0]
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
    residualsPoint,
    relativeErrorsPoint,
)
result

# %%
# Compute LARS method
transformation = ot.DistributionTransformation(im.inputDistribution, basis.getMeasure())
standard_input = transformation(input_sample)

# %%
# Initialisation
list_of_active_functions = [0]
coefficients = output_sample.computeMean()
residuals = ot.Sample(output_sample)
sample_size = standard_input.getSize()
maximum_basis_dimension = 10  # Set by the user
for i in range(maximum_basis_dimension - 1):
    # Find candidate with maximum absolute correlation with the residual
    print(f"Current active indices = {list_of_active_functions}")
    maximum_absolute_residual = 0.0
    best_basis_function_index = None
    for j in range(maximum_basis_dimension):
        if j in list_of_active_functions:
            continue
        current_basis_function = basis.build(j)
        basis_function_value = current_basis_function(standard_input)
        # What if the residuals has (output) dimension > 1 ?
        current_absolute_correlation = abs(
            residuals.asPoint().dot(basis_function_value.asPoint())
        )
        if current_absolute_correlation > maximum_absolute_residual:
            best_basis_function_index = j
            maximum_absolute_residual = current_absolute_correlation
    print(
        f"Best index = {best_basis_function_index} "
        f"with max. abs. corr. = {maximum_absolute_residual:.4f}"
    )
    # Add the best candidate to the active set
    list_of_active_functions.append(best_basis_function_index)
    # Update the coefficients
    functions = [basis.build(i) for i in list_of_active_functions]
    designProxy = ot.DesignProxy(standard_input, functions)
    leastSquaresMethod = ot.LeastSquaresMethod.Build(
        leastSquaresMethodName, designProxy, range(len(list_of_active_functions))
    )
    outputDimension = output_sample.getDimension()
    coefficients = ot.Sample(len(list_of_active_functions), outputDimension)
    for j in range(outputDimension):
        coeffsJ = leastSquaresMethod.solve(output_sample.getMarginal(j).asPoint())
        for i in range(len(list_of_active_functions)):
            coefficients[i, j] = coeffsJ[i]


# Réserve
# Create the result
# The physical model is unknown in this case ...
physicalModel = ot.Function()
# ... which implies that the composed model is unknown in this case
composedModel = ot.Function()
residualsPoint = [-1.0]
relativeErrorsPoint = [-1.0]
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
    residualsPoint,
    relativeErrorsPoint,
)
result

# %%
