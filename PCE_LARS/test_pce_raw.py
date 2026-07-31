"""Implements the polynomial chaos expansion algorithm in Python.

The goal of this script is to be used as a starting point of PCE
experiments, e.g. Orthogonal Matching Pursuit.
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
experiment = ot.MonteCarloExperiment(im.inputDistribution, sample_size)
input_sample, wX = experiment.generateWithWeights()
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
    leastSquaresMethodName, designProxy, wX, range(len(indices))
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
# TODO : Extend MetaModelValidation with weights 
# (see https://github.com/openturns/openturns/issues/2722)
experiment = ot.MonteCarloExperiment(im.inputDistribution, 100)
input_test = experiment.generate()
output_test = im.model(input_test)
meta_model = result.getMetaModel()
validation = ot.MetaModelValidation(output_test, meta_model(input_test))
print(f"Q2 = {validation.computeR2Score()[0]:.15f}")

