# -*- coding: utf-8 -*-
"""
Dans ce script, on mesure la performance des moindres carrés avec ou
sans sélection de modèle LARS.
"""

# %%
import openturns as ot
from matplotlib import pylab as plt
import time
import os
import numpy as np
import tqdm
import otbenchmark as otb
from openturns.usecases import ishigami_function


# %%
def computePCERMSE(result, inputTest, outputTest):
    metamodel = result.getMetaModel()
    predictions = metamodel(inputTest)
    validation = ot.MetaModelValidation(predictions, outputTest)
    fvu = 1.0 - max(0.0, validation.computeR2Score()[0])
    return fvu


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
def benchmarkPCETimeVsBasisSize(
    model,
    distribution,
    sampleSize,
    maximumElapsedTime=2.0,
    maximumNumberOfIterations=10,
    figsize=(4.0, 3.0),
    point_style_sparse="o",
    point_style_full="s",
    color_sparse="tab:blue",
    color_full="tab:orange",
    verbose=False,
):
    if verbose:
        print(f"Dimension = {distribution.getDimension()}")
        print(f"Sample size = {sampleSize}")
        print(f"Maximum number of iterations = {maximumNumberOfIterations}")
        print(f"Maximum elapsed time = {maximumElapsedTime}")
    # Train
    inputTrain = distribution.getSample(sampleSize)
    outputTrain = model(inputTrain)
    # Test
    testSampleSize = 1000
    inputTest = distribution.getSample(testSampleSize)
    outputTest = model(inputTest)
    #
    inputDimension = distribution.getDimension()
    listOfMarginals = [distribution.getMarginal(i) for i in range(inputDimension)]
    multivariateBasis = ot.OrthogonalProductPolynomialFactory(listOfMarginals)
    #
    listOfBasisSize = np.linspace(1.0, sampleSize, maximumNumberOfIterations)
    listOfBasisSize = [int(i) for i in listOfBasisSize]
    # Sparse PCE
    if verbose:
        print("+ Sparse PCE")
    listOfBasisSizeSparsePCE = []
    listOfFVUSparsePCE = []
    listOfElapsedTimeSparsePCE = []
    totalDegree = 0
    for iteration in tqdm.tqdm(range(maximumNumberOfIterations)):
        t1 = time.time()
        basisSize = max(1, int(listOfBasisSize[iteration] / 2))
        result = ComputeSparseLeastSquaresFunctionalChaos(
            inputTrain,
            outputTrain,
            multivariateBasis,
            basisSize,
            distribution,
            sparse=True,
        )
        t2 = time.time()
        elapsed_time = t2 - t1
        listOfBasisSizeSparsePCE.append(basisSize)
        listOfElapsedTimeSparsePCE.append(elapsed_time)
        fvu = computePCERMSE(result, inputTest, outputTest)
        listOfFVUSparsePCE.append(fvu)

        print(
            f"Iter={iteration}/{maximumNumberOfIterations}, "
            f"Sparse PCE, elapsed = {elapsed_time:.2f} (s), "
            f"Basis size = {basisSize}",
            f"FVU = {fvu:.2f}",
            flush=True,
        )
        if elapsed_time > maximumElapsedTime:
            break

    # Full PCE
    if verbose:
        print("+ Full PCE")
    totalDegree = 0
    listOfBasisSizeFullPCE = []
    listOfFVUFullPCE = []
    listOfElapsedTimeFullPCE = []
    for iteration in tqdm.tqdm(range(maximumNumberOfIterations)):
        totalDegree += 1
        t1 = time.time()
        basisSize = listOfBasisSize[iteration]
        result = ComputeSparseLeastSquaresFunctionalChaos(
            inputTrain,
            outputTrain,
            multivariateBasis,
            basisSize,
            distribution,
            sparse=False,
        )

        t2 = time.time()
        elapsed_time = t2 - t1
        listOfBasisSizeFullPCE.append(basisSize)
        listOfElapsedTimeFullPCE.append(elapsed_time)
        fvu = computePCERMSE(result, inputTest, outputTest)
        listOfFVUFullPCE.append(fvu)
        print(
            f"Iter={iteration}/{maximumNumberOfIterations}, "
            f"Full PCE, elapsed = {elapsed_time:.2f} (s), "
            f"Basis size = {basisSize}",
            f"FVU = {fvu:.2f}",
            flush=True,
        )
        if elapsed_time > maximumElapsedTime:
            break

    #
    if verbose:
        print("+ Plot elapsed time vs basis size")
    inputDimension = distribution.getDimension()
    # Plot Elapsed time
    figElapsedTimeVsBasisSize = plt.figure(figsize=figsize)
    plt.plot(
        listOfBasisSizeFullPCE,
        listOfElapsedTimeFullPCE,
        point_style_full,
        label="Full",
        color=color_full,
    )
    plt.plot(
        listOfBasisSizeSparsePCE,
        listOfElapsedTimeSparsePCE,
        point_style_sparse,
        label="Sparse",
        color=color_sparse,
    )
    plt.title(f"PCE, d={inputDimension}, n={sampleSize}")
    plt.xlabel("Basis dimension")
    plt.ylabel("Elapsed time (s)")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    # Plot FVU
    figElapsedTimeVsFVU = plt.figure(figsize=figsize)
    plt.plot(
        listOfFVUFullPCE,
        listOfElapsedTimeFullPCE,
        point_style_full,
        label="Full",
        color=color_full,
    )
    plt.plot(
        listOfFVUSparsePCE,
        listOfElapsedTimeSparsePCE,
        point_style_sparse,
        label="Sparse",
        color=color_sparse,
    )
    plt.xscale("log")
    plt.title(f"PCE, d={inputDimension}, n={sampleSize}")
    plt.xlabel("FVU")
    plt.ylabel("Elapsed time (s)")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    return figElapsedTimeVsBasisSize, figElapsedTimeVsFVU


# %%
ot.Log.Show(ot.Log.NONE)

# %%
sampleSize = 1000
benchmarkProblemList = otb.SensitivityBenchmarkProblemList()

for dimension in [10, 15, 20]:
    problem = otb.GSobolSensitivity(list(range(dimension)))
    problem.name = f"GSobol{dimension}"
    benchmarkProblemList.append(problem)
#
# Shortcut (to fine tune the benchmark)
# benchmarkProblemList = [otb.IshigamiSensitivity()]
#
numberOfProblems = len(benchmarkProblemList)
maximumElapsedTime = 2.0
maximumNumberOfIterations = 20
#
# %%
print("+ Mesure de la performance en fonction du degré")
print(f"n = {sampleSize}")
print(f"Nombre de problèmes = {numberOfProblems}")
print(f"Maximum elapsed time = {maximumElapsedTime:.2f} (s)")
print(f"Maximum number of iterations = {maximumNumberOfIterations}")
for i in range(numberOfProblems):
    print(f"Problem {i+1}/{numberOfProblems}...", flush=True)
    problem = benchmarkProblemList[i]
    name = problem.getName()
    distribution = problem.getInputDistribution()
    model = problem.getFunction()
    base_filename = str(name)
    for c in [" ", ".", "'", "-"]:
        base_filename = base_filename.replace(c, "")
    print(f"name = {name}")
    dimension = problem.getInputDistribution().getDimension()
    figElapsedTimeVsBasisSize, figElapsedTimeVsFVU = benchmarkPCETimeVsBasisSize(
        model,
        distribution,
        sampleSize,
        maximumElapsedTime=maximumElapsedTime,
        maximumNumberOfIterations=maximumNumberOfIterations,
        verbose=True,
    )
    # Sauve Basis Size
    ax = figElapsedTimeVsBasisSize.gca()  # Récupère l'axe actuel (Get Current Axis)
    ax.set_title(f"{name}, d={distribution.getDimension()}, n={sampleSize}")
    for extension in [".pdf", ".png"]:
        filename = f"figures/benchmark_sparse_least_squares_{base_filename}_BasisSize{extension}"
        print(f"Write on {filename}...")
        _ = figElapsedTimeVsBasisSize.savefig(
            os.path.join(filename), bbox_inches="tight"
        )

    # Sauve RMSE
    ax = figElapsedTimeVsFVU.gca()  # Récupère l'axe actuel (Get Current Axis)
    ax.set_title(f"{name}, d={distribution.getDimension()}, n={sampleSize}")
    for extension in [".pdf", ".png"]:
        filename = (
            f"figures/benchmark_sparse_least_squares_{base_filename}_FVU{extension}"
        )
        print(f"Write on {filename}...")
        _ = figElapsedTimeVsFVU.savefig(os.path.join(filename), bbox_inches="tight")
    #
    _ = plt.show()


# %%
