#! /usr/bin/env python

import openturns as ot
from openturns.usecases import flood_model

sampleSize = 500
totalDegree = 7

fm = flood_model.FloodModel()
inputDescription = fm.model.getInputDescription()
marginals = [fm.distribution.getMarginal(i) for i in range(fm.dim)]
basis = ot.OrthogonalProductPolynomialFactory(marginals)
inputSample = fm.distribution.getSample(sampleSize)
outputSample = fm.model(inputSample)
selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory()
projectionStrategy = ot.LeastSquaresStrategy(selectionAlgorithm)
enumerateFunction = basis.getEnumerateFunction()
basisSize = enumerateFunction.getBasisSizeFromTotalDegree(totalDegree)
adaptiveStrategy = ot.FixedStrategy(basis, basisSize)
algo = ot.FunctionalChaosAlgorithm(
    inputSample, outputSample, fm.distribution, adaptiveStrategy, projectionStrategy
)
algo.run()
result = algo.getResult()
print("+ FCResult:")
print(result)

distribution = result.getDistribution()
print("+ distribution:")
print(distribution)

orthogonalBasis = result.getOrthogonalBasis()
print("+ orthogonalBasis:")
print(orthogonalBasis)

measure = orthogonalBasis.getMeasure()
print("+ Measure:")
print(measure)


implementation = orthogonalBasis.getImplementation()
polynomialCollection = implementation.getPolynomialFamilyCollection()
for i in range(len(polynomialCollection)):
    print("i = ", i)
    marginalFamily = polynomialCollection[i]
    marginalPolynomial = marginalFamily.getImplementation()
    className = marginalPolynomial.getClassName()
    print("  ClassName=", className)
    if marginalPolynomial.ANALYSIS == 1:
        print("  Analysis")
    elif marginalPolynomial.PROBABILITY == 1:
        print("  Probability")
    else:
        raise ValueError("Polynomial is neither analysis nor probability.")
    basicPolynomial = marginalFamily.getImplementation()
