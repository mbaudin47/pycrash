# -*- coding: utf-8 -*-
"""
Dans ce script, on mesure la performance des moindres carrés avec ou
sans sélection de modèle LARS.
"""

# %%
import openturns as ot
from matplotlib import pylab as plt
from openturns.usecases import ishigami_function
import time
import os
import numpy as np


# %%
def ComputeSparseLeastSquaresFunctionalChaos(
    inputTrain,
    outputTrain,
    multivariateBasis,
    basisSize,
    distribution,
    sparse=True,
):
    """
    Create a sparse polynomial chaos based on least squares.

    * Uses the enumerate rule in multivariateBasis.
    * Uses the LeastSquaresStrategy to compute the coefficients based on
      least squares.
    * Uses LeastSquaresMetaModelSelectionFactory to use the LARS selection method.
    * Uses FixedStrategy in order to keep all the coefficients that the
      LARS method selected.

    Parameters
    ----------
    inputTrain : ot.Sample
        The input design of experiments.
    outputTrain : ot.Sample
        The output design of experiments.
    multivariateBasis : ot.Basis
        The multivariate chaos basis.
    basisSize : int
        The size of the function basis.
    distribution : ot.Distribution.
        The distribution of the input variable.
    sparse: bool
        If True, create a sparse PCE.

    Returns
    -------
    result : ot.PolynomialChaosResult
        The estimated polynomial chaos.
    """
    if sparse:
        selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory()
    else:
        selectionAlgorithm = ot.PenalizedLeastSquaresAlgorithmFactory()
    projectionStrategy = ot.LeastSquaresStrategy(
        inputTrain, outputTrain, selectionAlgorithm
    )
    adaptiveStrategy = ot.FixedStrategy(multivariateBasis, basisSize)
    chaosAlgorithm = ot.FunctionalChaosAlgorithm(
        inputTrain, outputTrain, distribution, adaptiveStrategy, projectionStrategy
    )
    chaosAlgorithm.run()
    chaosResult = chaosAlgorithm.getResult()
    return chaosResult


# %%
ot.Log.Show(ot.Log.NONE)

# %%
im = ishigami_function.IshigamiModel()
sampleSize = 1000
inputTrain = im.distribution.getSample(sampleSize)
outputTrain = im.model(inputTrain)
multivariateBasis = ot.OrthogonalProductPolynomialFactory([im.X1, im.X2, im.X3])
inputDimension = im.distribution.getDimension()

# %%
print("+ Mesure de la performance en fonction du degré")
maximumElapsedTime = 2.0
maximumNumberOfIterations = 10

# %%
listOfBasisSize = np.linspace(1.0, sampleSize, maximumNumberOfIterations)
listOfBasisSize = [int(i) for i in listOfBasisSize]

# %%
# Sparse PCE
listOfBasisSizeSparsePCE = []
listOfElapsedTimeSparsePCE = []
totalDegree = 0
for iteration in range(maximumNumberOfIterations):
    t1 = time.time()
    basisSize = listOfBasisSize[iteration]
    result = ComputeSparseLeastSquaresFunctionalChaos(
        inputTrain,
        outputTrain,
        multivariateBasis,
        basisSize,
        im.distribution,
        sparse=True,
    )
    t2 = time.time()
    elapsed_time = t2 - t1
    listOfBasisSizeSparsePCE.append(basisSize)
    listOfElapsedTimeSparsePCE.append(elapsed_time)
    print(
        f"Iter={iteration}/{maximumNumberOfIterations}, "
        f"Sparse PCE, elapsed = {elapsed_time:.2f} (s), "
        f"Basis size = {basisSize}", flush=True
    )
    if elapsed_time > maximumElapsedTime:
        break


# %%
# Full PCE
totalDegree = 0
listOfBasisSizeFullPCE = []
listOfElapsedTimeFullPCE = []
for iteration in range(maximumNumberOfIterations):
    totalDegree += 1
    t1 = time.time()
    basisSize = listOfBasisSize[iteration]
    result = ComputeSparseLeastSquaresFunctionalChaos(
        inputTrain,
        outputTrain,
        multivariateBasis,
        basisSize,
        im.distribution,
        sparse=False,
    )

    t2 = time.time()
    elapsed_time = t2 - t1
    listOfBasisSizeFullPCE.append(basisSize)
    listOfElapsedTimeFullPCE.append(elapsed_time)
    print(
        f"Iter={iteration}/{maximumNumberOfIterations}, "
        f"Full PCE, elapsed = {elapsed_time:.2f} (s), "
        f"Basis size = {basisSize}", flush=True
    )
    if elapsed_time > maximumElapsedTime:
        break

# %%
print("+ Plot elapsed time vs degré")
fig = plt.figure(figsize=(4.0, 3.0))
plt.plot(listOfBasisSizeFullPCE, listOfElapsedTimeFullPCE, "o", label="Full")
plt.plot(listOfBasisSizeSparsePCE, listOfElapsedTimeSparsePCE, "o", label="Sparse")
plt.title(f"PCE, d={inputDimension}, n={sampleSize}")
plt.xlabel("Basis dimension")
plt.ylabel("Elapsed time (s)")
plt.yscale("log")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
plt.savefig(
    os.path.join("benchmark_sparse_least_squares.pdf"), bbox_inches="tight"
)

# %%
