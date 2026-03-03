#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On recherche la pirte perturbation pour le système d'équations linéaires.
"""

import openturns as ot
import openturns.viewer as otv
import otcalibration as otc
import numpy as np

size = 10
inputObservations = ot.Sample([[0.5 + i] for i in range(size)])

inVars = ["a", "b", "x"]
formulas = ["a + exp(b * x)"]
g = ot.SymbolicFunction(inVars, formulas)

# Set mu
parameter_true = [2.8, 0.5]
params = [0, 1]
model = ot.ParametricFunction(g, params, parameter_true)
model.setOutputDescription(["y"])

# Set epsilon
sigmaNoiseObservation = 5.0
observationOutputNoise = ot.Normal(0.0, sigmaNoiseObservation)

# Set B
sigma_a = 10.0
sigma_b = 1.0
parameterCovariance = ot.CovarianceMatrix([[sigma_a ** 2, 0.0], [0.0, sigma_b ** 2]])
print(parameterCovariance)

# Set R
errorCovariance = ot.CovarianceMatrix([[3.0]])

# candidate = ot.Point([5.0, 0.7])
candidate = (1.0 + 0.1) * ot.Point(parameter_true)
print("candidate=", candidate)

K = otc.computeKalmanMatrix(
    model,
    inputObservations,
    candidate,
    parameterCovariance,
    errorCovariance,
    verbose=True,
)
print("K=")
print(K)


def vectorMaxNorm1(A):
    """
    Calcule le vecteur x qui maximise
    la norme 1 de ||A*x||/||x||
    """
    k = np.argmax(np.sum(np.abs(A), 0))
    n = A.shape[1]
    x = np.zeros((n, 1))
    x[k] = 1.0
    return x


def vectorMaxNormInf(A):
    """
    Calcule le vecteur x qui maximise
    la norme infinie de ||A*x||/||x||
    """
    if type(A) != np.ndarray:
        raise TypeError("A numpy array is expected.")
    k = np.argmax(np.sum(np.abs(A), 1))
    n = A.shape[1]
    x = np.zeros((n, 1))
    #
    indices = np.where(A[k, :] > 0)[0]
    x[indices] = 1.0
    #
    indices = np.where(A[k, :] < 0)[0]
    x[indices] = -1.0
    return x


# Compute worst case vector x
Karray = np.array(K)
x_inf = vectorMaxNormInf(Karray)
print("x_inf =")
print(x_inf)

# Set noise
amplitude = 5.0
epsilon = amplitude * ot.Point(x_inf.flatten())

# Compute true observations
outputModel = model(inputObservations)

# Set observations
observationErrorSample = ot.Sample([[v] for v in epsilon])
outputObservations = outputModel + observationErrorSample

graph = model.draw(0.5, 9.5)
graph.setLegends(["True model"])
cloud = ot.Cloud(inputObservations, outputObservations)
cloud.setLegend("Observations")
graph.add(cloud)
graph.setLegendPosition("topleft")
graph.setColors(ot.Drawable_BuildDefaultPalette(2))
graph.setTitle("Exponential model.")
#
view = otv.View(graph)


def compareTrueAndEstimateModel(
    parameter_true, parameter_estimate, inputObservations, outputObservations
):
    # 1. True model
    model.setParameter(parameter_true)
    graph = model.draw(0.5, 9.5)
    graph.setLegends(["True"])
    # 2. Obs
    cloud = ot.Cloud(inputObservations, outputObservations)
    cloud.setLegend("Obs.")
    graph.add(cloud)
    # 4.
    model.setParameter(parameter_estimate)
    curve = model.draw(0.5, 9.5)
    curve.setLegends(["Calib."])
    graph.add(curve)
    graph.setYTitle("y")
    #
    graph.setLegendPosition("topleft")
    graph.setColors(ot.Drawable_BuildDefaultPalette(4))
    return graph


(
    thetaStarCholesky,
    covarianceThetaStarCholesky,
) = otc.gaussianLinearCalibrationFromCholesky(
    model,
    inputObservations,
    outputObservations,
    candidate,
    parameterCovariance,
    errorCovariance,
    verbose=True,
)
print("thetaStarCholesky=", thetaStarCholesky)

thetaStarKalman, covarianceThetaStarKalman = otc.gaussianLinearCalibrationFromKalman(
    model,
    inputObservations,
    outputObservations,
    candidate,
    parameterCovariance,
    errorCovariance,
    verbose=True,
)

print("thetaStarKalman=", thetaStarKalman)

print("Diff=", thetaStarKalman - thetaStarCholesky)

graph_Chol = compareTrueAndEstimateModel(
    parameter_true, thetaStarCholesky, inputObservations, outputObservations
)
graph_Chol.setTitle("Calibration from Cholesky")

graph_Kalm = compareTrueAndEstimateModel(
    parameter_true, thetaStarKalman, inputObservations, outputObservations
)
graph_Kalm.setTitle(r".$\qquad$ Calibration from Kalman")

grid = ot.GridLayout(1, 2)
grid.setGraph(0, 0, graph_Chol)
grid.setGraph(0, 1, graph_Kalm)
view = otv.View(grid, figure_kw={"figsize": (9.0, 2.5)})
