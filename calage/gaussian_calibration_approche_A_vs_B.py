# -*- coding: utf-8 -*-
"""
Compare les approches A et B sur un exemple.
Adapté de t_GaussianLinearCalibration_std.py.
"""
#! /usr/bin/env python

from __future__ import print_function
import openturns as ot
from openturns.testing import assert_almost_equal

ot.TESTPREAMBLE()
ot.PlatformInfo.SetNumericalPrecision(5)

#
print("+ Approche B")

"""
Test the calibration of an exponential function with 3 parameters.
Test the three decompositions (QR, SVD, Cholesky) of the least
squares problem.
Test the local and global error covariances.
"""
m = 10
inputObservations = [[0.5 + i] for i in range(m)]
print("Observed inputs : ")
print(inputObservations)

inVars = ["a", "b", "c", "inputObservations"]
formulas = ["a + b * exp(c * inputObservations)", "(a * inputObservations^2 + b) / (c + inputObservations^2)"]
g = ot.SymbolicFunction(inVars, formulas)

trueParameter = [2.8, 1.2, 0.5]
print("True parameter =", trueParameter)

calibratedIndices = [0, 1, 2]
parametricModel = ot.ParametricFunction(g, calibratedIndices, trueParameter)
outputObservations = parametricModel(inputObservations)
outputObservations += ot.Normal([0.0] * 2, [0.05] * 2, ot.IdentityMatrix(2)).getSample(m)
print("Observed outputs :")
print(outputObservations)

# See the parametric model in action
oneSingleInputObservation = [5.5]
oneSingleOutputObservation = parametricModel(oneSingleInputObservation)
print("Single observed output :")
print(oneSingleOutputObservation)

candidate = [1.0] * 3
print("Initial parameter =", candidate)

priorCovariance = ot.CovarianceMatrix(3)
for i in range(3):
    priorCovariance[i, i] = 3.0 + (1.0 + i) * (1.0 + i)
    for j in range(i):
        priorCovariance[i, j] = 1.0 / (1.0 + i + j)
errorCovariance = ot.CovarianceMatrix(2)
for i in range(2):
    errorCovariance[i, i] = 2.0 + (1.0 + i) * (1.0 + i)
    for j in range(i):
        errorCovariance[i, j] = 1.0 / (1.0 + i + j)
globalErrorCovariance = ot.CovarianceMatrix(2 * m)
for i in range(2 * m):
    globalErrorCovariance[i, i] = 2.0 + (1.0 + i) * (1.0 + i)
    for j in range(i):
        globalErrorCovariance[i, j] = 1.0 / (1.0 + i + j)

method = "SVD"
print("method=", method)
# 1. Check with local error covariance
print("Local error covariance")
algo = ot.GaussianLinearCalibration(
    parametricModel, inputObservations, outputObservations, candidate, priorCovariance, errorCovariance, method
)
algo.run()
calibrationResult = algo.getResult()

thetaMAP = calibrationResult.getParameterMAP()
print("MAP=", thetaMAP)

# 2. Check with global error covariance
print("Global error covariance")
algo = ot.GaussianLinearCalibration(
    parametricModel, inputObservations, outputObservations, candidate, priorCovariance, globalErrorCovariance, method
)
algo.run()
calibrationResult = algo.getResult()

thetaMAP = calibrationResult.getParameterMAP()
print("MAP=", thetaMAP)

#
print("+ Approche A : prototype")
inputIndices = [3]
referenceInputValue = [5.5]
parametricModel = ot.ParametricFunction(g, inputIndices, referenceInputValue)

# See the parametric model in action
oneSingleOutputObservation = parametricModel(candidate)
print("Single observed output :")
print(oneSingleOutputObservation)

if False:
    """
    TypeError: InvalidArgumentException : Error: expected a model of parameter 
    dimension=3, got parameter dimension=1
    """
    algo = ot.GaussianLinearCalibration(
        parametricModel, inputObservations, outputObservations, candidate, priorCovariance, errorCovariance, method
    )
