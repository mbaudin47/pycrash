#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This shows how to create a FunctionalChaosResult.
This is a simplified version of LeastSquaresFCE.

Authors : Sofiane Haddad, Régis Lebrun (Airbus), Michaël Baudin (EDF)

This is a reply to:

https://openturns.discourse.group/t/problem-with-sensitivity-analysis-using-functionalchaosresult/378

"""


# %%
import openturns as ot
from math import pi
from time import time

# %%
a = 7.0
b = 0.1
inputVariables = ["xi1", "xi2", "xi3"]
formula = [
    "sin(xi1) + ("
    + str(a)
    + ") * (sin(xi2)) ^ 2 + ("
    + str(b)
    + ") * xi3^4 * sin(xi1)"
]
model = ot.SymbolicFunction(inputVariables, formula)

# %%
# Create the input distribution
dimension = len(inputVariables)
distribution = ot.ComposedDistribution([ot.Uniform(-pi, pi)] * dimension)

# %%
# Create the orthogonal basis
enumerateFunction = ot.LinearEnumerateFunction(dimension)
basis = ot.OrthogonalProductPolynomialFactory(
    [ot.LegendreFactory()] * dimension, enumerateFunction
)

# %%
# Create the input/output database
size = 10000
input_sample = distribution.getSample(size)
output_sample = model(input_sample)
totalDegree = 5
strataIndex = enumerateFunction.getMaximumDegreeStrataIndex(totalDegree)
maximumBasisSize = enumerateFunction.getStrataCumulatedCardinal(strataIndex)
transformation = ot.DistributionTransformation(distribution, basis.getMeasure())

# %%
# Create a polynomial chaos expansion, step-by-step
standard_input = transformation(input_sample)
indices = ot.Indices(maximumBasisSize)
indices.fill()
functions = [basis.build(i) for i in indices]
designProxy = ot.DesignProxy(standard_input, functions)
leastSquaresMethod="SVD"
leastSquaresMethod = ot.LeastSquaresMethod.Build(
    leastSquaresMethod, designProxy, indices
)
outputDimension = output_sample.getDimension()
coefficients = ot.Sample(maximumBasisSize, outputDimension)
for j in range(outputDimension):
    coeffsJ = leastSquaresMethod.solve(output_sample.getMarginal(j).asPoint())
    for i in range(maximumBasisSize):
        coefficients[i, j] = coeffsJ[i]
# Create the result
# The physical model is unknown in this case ...
physicalModel = ot.Function()
# ... which implies that the composed model is unknown in this case
composedModel = ot.Function()
residualsPoint = [-1.0]
relativeErrorsPoint = [-1.0]
"""
OT::FunctionalChaosResult::FunctionalChaosResult(
    OT::Sample const &, 1
    OT::Sample const &, 2
    OT::Distribution const &, 3
    OT::Function const &, 4
    OT::Function const &, 5
    OT::OrthogonalBasis const &, 6
    OT::Indices const &, 7
    OT::Sample const &, 8
    OT::FunctionalChaosResult::FunctionCollection const &, 9
    OT::Point const &, 10
    OT::Point const &) 11
"""
result = ot.FunctionalChaosResult(
    input_sample,  # 1
    output_sample, # 2
    distribution,  # 3
    transformation,  # 4
    transformation.inverse(),  # 5
    basis,  # 6
    indices,  # 7
    coefficients,  # 8
    functions,  # 9
    residualsPoint,  # 10
    relativeErrorsPoint,  # 11
)
result

# %%
