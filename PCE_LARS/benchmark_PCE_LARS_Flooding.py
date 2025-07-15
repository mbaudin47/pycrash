"""Shows that LARS can be unstable when the sample size is small.
References
----------
https://github.com/openturns/openturns/issues/2495
"""

# %%
import openturns as ot
import openturns.viewer as otv
import numpy as np

# See https://github.com/openturns/openturns/issues/2495
# ot.ResourceMap.SetAsScalar("LeastSquaresMetaModelSelection-MaximumErrorFactor", 10.0)
# ot.ResourceMap.SetAsScalar("LeastSquaresMetaModelSelection-MaximumError", 10.0)
# ot.ResourceMap.SetAsScalar("LeastSquaresMetaModelSelection-ErrorTreshold", 10.0)

# %%
sampleSizeTrain = 10  # The size of the train sample
totalDegree = 4  # The total degree of each PCE expert
sparse = True  # Set to True to use a sparse PCE

# Load the cantilever beam model
physicalModel = ot.SymbolicFunction(
    ["Q", "Ks"], ["H"], "H := (Q / (Ks * 9.486))^(3.0 / 5.0)"
)
Q = ot.TruncatedDistribution(
    ot.Gumbel(558.0, 1013.0), 0.0, ot.TruncatedDistribution.LOWER
)
Ks = ot.TruncatedDistribution(
    ot.Normal(30.0, 7.5), 10.0, ot.TruncatedDistribution.LOWER
)
distribution = ot.ComposedDistribution([Q, Ks])


# %%
magicSeed = 4
# for magicSeed in range(100):

# %%
# ot.ResourceMap.SetAsScalar("LeastSquaresMetaModelSelection-MaximumErrorFactor", np.inf)
# ot.ResourceMap.SetAsScalar("LeastSquaresMetaModelSelection-MaximumError", np.inf)
# ot.ResourceMap.SetAsScalar("LeastSquaresMetaModelSelection-ErrorThreshold", 0.0)


# %%
ot.RandomGenerator.SetSeed(magicSeed)

#
inputTrain = distribution.getSample(sampleSizeTrain)
outputTrain = physicalModel(inputTrain)

# Create orthogonal basis
inputDimension = distribution.getDimension()
listOfMarginals = [distribution.getMarginal(i) for i in range(inputDimension)]
multivariateBasis = ot.OrthogonalProductPolynomialFactory(listOfMarginals)
enumerateFunction = multivariateBasis.getEnumerateFunction()
basisDimension = enumerateFunction.getBasisSizeFromTotalDegree(totalDegree)
print(f"basisDimension = {basisDimension}")

# %%
# Train a PCE
if sparse:
    basisSequenceFactory = ot.LARS()
    fittingAlgorithm = ot.CorrectedLeaveOneOut()
    # k_parameter = int(inputTrain.getSize() / 2)
    # fittingAlgorithm = ot.KFold(k_parameter)
    selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory(
        basisSequenceFactory, fittingAlgorithm
    )
else:
    selectionAlgorithm = ot.PenalizedLeastSquaresAlgorithmFactory()
projectionStrategy = ot.LeastSquaresStrategy(selectionAlgorithm)
enumerateFunction = multivariateBasis.getEnumerateFunction()
maximumNumberOfCoefficients = enumerateFunction.getStrataCumulatedCardinal(totalDegree)
adaptiveStrategy = ot.FixedStrategy(multivariateBasis, maximumNumberOfCoefficients)
chaosAlgorithm = ot.FunctionalChaosAlgorithm(
    inputTrain, outputTrain, distribution, adaptiveStrategy, projectionStrategy
)
chaosAlgorithm.run()
result = chaosAlgorithm.getResult()
print(result)

# %%
metaModel = result.getMetaModel()
validation = ot.MetaModelValidation(outputTrain, metaModel(inputTrain))
r2Score = validation.computeR2Score()
print(magicSeed, r2Score)

graph = validation.drawValidation()
otv.View(graph)

# %%
graph = result.drawErrorHistory()
graph.setTitle(f"n = {sampleSizeTrain}")
graph

# %%
graph = result.drawSelectionHistory()
graph.setTitle(f"n = {sampleSizeTrain}")
graph.setLegendPosition("upper left")
graph.setLegendCorner((1.0, 1.0))
graph

# %%
