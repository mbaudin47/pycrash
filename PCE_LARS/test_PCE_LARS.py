"""Implement the LARS algorithm in Python"""

# %%
import openturns as ot
from openturns.usecases import ishigami_function

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
input_test = im.inputDistribution.getSample(1000)
output_test = im.model(input_test)
meta_model = result.getMetaModel()
validation = ot.MetaModelValidation(output_test, meta_model(input_test))
print(f"Q2 = {validation.computeR2Score()[0]:.15f}")

# %%
# 2. Compute OMP method
sample_size = standard_input.getSize()
transformation = ot.DistributionTransformation(im.inputDistribution, basis.getMeasure())
standard_input = transformation(input_sample)
# Create a list of functions
functions = [basis.build(i) for i in range(maximum_basis_dimension)]
designProxy = ot.DesignProxy(standard_input, functions)
# Initialisation
list_of_active_functions = [0]  # Initialize with constant basis
leastSquaresMethod = ot.LeastSquaresMethod.Build(
    leastSquaresMethodName, designProxy, list_of_active_functions
)
residuals = output_sample.asPoint()
# Update residuals
residuals -= ot.Point(sample_size, output_sample.computeMean()[0])
# TODO: subtract the sample mean
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
        current_basis_function = basis.build(j)
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
    cloo = ot.CorrectedLeaveOneOut()
    score = cloo.run(
        standard_input,
        output_sample,
        ot.Point(sample_size, 1) / sample_size,
        functions,
        list_of_active_functions,
    )
    print(f"  CLOO score = {score:.4e}")

coefficientSample = ot.Sample.BuildFromPoint(coefficients)

# Réserve
# Create the result
functions = [basis.build(i) for i in list_of_active_functions]
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
    list_of_active_functions,
    coefficientSample,
    functions,
    residualsPoint,
    relativeErrorsPoint,
)
result

# %%
meta_model = result.getMetaModel()
validation = ot.MetaModelValidation(output_test, meta_model(input_test))
print(f"Q2 = {validation.computeR2Score()[0]:.15f}")

# %%
