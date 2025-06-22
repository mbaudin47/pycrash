# -*- coding: utf-8 -*-
"""
Dans ce script, on recherche le meilleur degré polynomial entre 1 et 10,
et la meilleure règle d'énumération.
On étude 2 variables de la décomposition en polynômes du chaos : plein ou creux.

Montre que la méthode de sélection de modèle peut être coûteuse lorsque 
la dimension de la base de fonctions augmente.
"""
# %%
import openturns as ot
import openturns.viewer as otv
from matplotlib import pylab as plt
import time
import numpy as np


# %%
def compute_sparse_least_squares_chaos(
    inputTrain, outputTrain, multivariateBasis, totalDegree, myDistribution, sparse=True
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
    totalDegree : int
        The total degree of the chaos polynomial.
    myDistribution : ot.Distribution.
        The distribution of the input variable.
    sparse : bool
        If True, then use the LARS algorithm to perform select best coefficients.
        Otherwise, compute and keep all coefficients.

    Returns
    -------
    result : ot.PolynomialChaosResult
        The estimated polynomial chaos.
    """
    if sparse:
        # Use LeastSquaresMetaModelSelection-DecompositionMethod to
        # select QR, SVD or Cholesky
        selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory()
    else:
        # Compute Gram matrix, always
        selectionAlgorithm = ot.PenalizedLeastSquaresAlgorithmFactory()
    projectionStrategy = ot.LeastSquaresStrategy(
        inputTrain, outputTrain, selectionAlgorithm
    )
    enumerateFunction = multivariateBasis.getEnumerateFunction()
    strataIndex = enumerateFunction.getMaximumDegreeStrataIndex(totalDegree)
    maximumBasisSize = enumerateFunction.getStrataCumulatedCardinal(strataIndex)
    adaptiveStrategy = ot.FixedStrategy(multivariateBasis, maximumBasisSize)
    chaosalgo = ot.FunctionalChaosAlgorithm(
        inputTrain, outputTrain, myDistribution, adaptiveStrategy, projectionStrategy
    )
    chaosalgo.run()
    result = chaosalgo.getResult()
    return result


# %%
# From https://raw.githubusercontent.com/openturns/otbenchmark/refs/heads/master/otbenchmark/_GSobolSensitivity.py
gsobol_dimension = 6
gsobol_a = [float(v) for v in np.linspace(0.0, 99.0, gsobol_dimension)]
gsobol_a


# %%
def GSobolModel(X):
    X = ot.Point(X)
    d = X.getDimension()
    Y = 1.0
    for i in range(d):
        Y *= (abs(4.0 * X[i] - 2.0) + gsobol_a[i]) / (1.0 + gsobol_a[i])
    return ot.Point([Y])


g_model = ot.PythonFunction(gsobol_dimension, 1, GSobolModel)
g_model.setOutputDescription(["Y"])

# %%
# Define the distribution
distributionList = [ot.Uniform(0.0, 1.0) for i in range(gsobol_dimension)]
distributionInput = ot.ComposedDistribution(distributionList)

# %%
ot.Log.Show(ot.Log.NONE)

# %%
sample_size = 1000
sample_input = distributionInput.getSample(sample_size)
sample_output = g_model(sample_input)

print("sample_input")
print(sample_input[:5])
print("sample_output")
print(sample_output[:5])

# %%
dimension_input = distributionInput.getDimension()
multivariateBasis = ot.OrthogonalProductPolynomialFactory(
    [distributionInput.getMarginal(i) for i in range(dimension_input)]
)
multivariateBasis

# %%
totalDegree = 2  # Polynomial degree
chaos_result = compute_sparse_least_squares_chaos(
    sample_input,
    sample_output,
    multivariateBasis,
    totalDegree,
    distributionInput,
)
metamodel = chaos_result.getMetaModel()

# %%
# Validation
sample_input = distributionInput.getSample(sample_size)
sample_output = g_model(sample_input)
validation = ot.MetaModelValidation(sample_output, metamodel(sample_input))
score_Q2 = validation.computeR2Score()[0]
print("Degree = %d" % (totalDegree))
print(f"Score Q2 = {score_Q2}")

# %%
print("+ Mesure de la performance en fonction du degré")
maximumElapsedTime = 120.0
maximumNumberOfIterations = 10
# Sparse PCE
listOfDegreesSparsePCE = []
listOfElapsedTimeSparsePCE = []
totalDegree = 0
for iteration in range(maximumNumberOfIterations):
    t1 = time.time()
    totalDegree += 1
    result = compute_sparse_least_squares_chaos(
        sample_input,
        sample_output,
        multivariateBasis,
        totalDegree,
        distributionInput,
        sparse=True,
    )
    t2 = time.time()
    elapsed_time = t2 - t1
    listOfDegreesSparsePCE.append(totalDegree)
    listOfElapsedTimeSparsePCE.append(elapsed_time)
    print(
        f"Iter={iteration}/{maximumNumberOfIterations}, Sparse PCE, "
        f"elapsed = {elapsed_time:.2f} (s), "
        f"Degree = {totalDegree}"
    )
    if elapsed_time > maximumElapsedTime:
        break

# %%
# Full PCE
totalDegree = 0
listOfDegreesFullPCE = []
listOfElapsedTimeFullPCE = []
for iteration in range(maximumNumberOfIterations):
    totalDegree += 1
    t1 = time.time()
    result = compute_sparse_least_squares_chaos(
        sample_input,
        sample_output,
        multivariateBasis,
        totalDegree,
        distributionInput,
        sparse=False,
    )
    t2 = time.time()
    elapsed_time = t2 - t1
    listOfDegreesFullPCE.append(totalDegree)
    listOfElapsedTimeFullPCE.append(elapsed_time)
    print(
        f"Iter={iteration}/{maximumNumberOfIterations}, Full PCE, "
        f"elapsed = {elapsed_time:.2f} (s), "
        f"Degree = {totalDegree}"
    )
    if elapsed_time > maximumElapsedTime:
        break

# %%
print("+ Plot elapsed time vs degré")
fig = plt.figure(figsize=(4.0, 3.0))
plt.plot(listOfDegreesFullPCE, listOfElapsedTimeFullPCE, "o", label="Full")
plt.plot(listOfDegreesSparsePCE, listOfElapsedTimeSparsePCE, "o", label="Sparse")
plt.title(f"Polynomial chaos expansion, n={sample_input.getSize()}")
plt.xlabel("Degree")
plt.ylabel("Elapsed time (s)")
plt.yscale("log")
_ = plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")


# %%
