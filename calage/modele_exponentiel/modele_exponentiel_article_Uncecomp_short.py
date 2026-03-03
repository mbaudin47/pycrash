#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On reproduit les calculs de l'article Uncecomp sur le modèle exponentiel.
"""

import openturns as ot
import openturns.viewer as otv
import otcalibration as otc
import pylab as pl
import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "10"

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
parameterCovariance = ot.CovarianceMatrix([[4.0, 0.5], [0.5, 7.0]])
print(parameterCovariance)

# Set R
errorCovariance = ot.CovarianceMatrix([[3.0]])

candidate = ot.Point([1.0, 1.0])
print("candidate=", candidate)

# Observations
size = 10
outputObservations = ot.Sample(size, 1)
outputObservations[0, 0] = 7.125
outputObservations[1, 0] = -1.414
outputObservations[2, 0] = 4.099
outputObservations[3, 0] = 14.58
outputObservations[4, 0] = 1.381
outputObservations[5, 0] = 20.19
outputObservations[6, 0] = 26.82
outputObservations[7, 0] = 52.52
outputObservations[8, 0] = 76.96
outputObservations[9, 0] = 122.4

# Plot
model.setParameter(parameter_true)
graph = model.draw(0.5, 9.5)
graph.setLegends(["True model"])
cloud = ot.Cloud(inputObservations, outputObservations)
cloud.setLegend("Observations")
graph.add(cloud)
graph.setLegendPosition("topleft")
graph.setColors(ot.Drawable_BuildDefaultPalette(2))
graph.setTitle("Exponential model.")
#
view = otv.View(graph, figure_kw={"figsize": (3.0, 2.0)})

# Plot
model.setParameter(parameter_true)
graph = model.draw(0.5, 9.5)
graph.setLegends(["True model"])
cloud = ot.Cloud(inputObservations, outputObservations)
cloud.setLegend("Observations")
graph.add(cloud)
model.setParameter(candidate)
curve = model.draw(0.5, 9.5)
curve.setLegends(["Candidate model"])
graph.add(curve)
graph.setLegendPosition("topleft")
graph.setColors(ot.Drawable_BuildDefaultPalette(3))
graph.setTitle("Exponential model.")
graph.setXTitle("$x$")
#
view = otv.View(graph, figure_kw={"figsize": (3.0, 2.0)})
figure = view.getFigure()
figure.savefig("exponentiel_true_vs_candidate.pdf", bbox_inches='tight')


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
    graph.setYTitle("$y$")
    graph.setXTitle("$x$")
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
graph_Kalm.setTitle(r"Calibration from Kalman")

grid = ot.GridLayout(1, 2)
grid.setGraph(0, 0, graph_Chol)
grid.setGraph(0, 1, graph_Kalm)
view = otv.View(grid, figure_kw={"figsize": (5.0, 2.0)})
pl.subplots_adjust(wspace=0.5)
figure = view.getFigure()
figure.savefig("exponentiel_C_vs_K.pdf", bbox_inches='tight')
