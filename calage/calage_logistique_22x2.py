#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:31:26 2020
logistic calibration 2 countries
@author: d16574 - Sanaa Zannane +Michaël Baudin (2021)

On considère un 1 échantillon de taille 22 en dimension 2.
"""

import openturns as ot
import numpy as np
import openturns.viewer as otv
import pylab as pl

# Functions + data + constants


def modele_une_population(t, t0, y0, a, b, m):
    y = m + a * y0 / (b * y0 + (a - b * y0) * np.exp(-a * (t - t0)))
    return y


def logisticModel(X):
    t = X[0]
    a_US = X[1]
    c_US = X[2]
    a_FR = X[3]
    c_FR = X[4]
    m = X[5]

    t0 = 1790.0
    y0_US = 3.9e6
    y0_FR = 28.1e6
    b_US = np.exp(c_US)
    b_FR = np.exp(c_FR)

    scaled_m = m * 1.0e6

    y = ot.Point(2)
    y[0] = modele_une_population(t, t0, y0_US, a_US, b_US, scaled_m)
    y[1] = modele_une_population(t, t0, y0_FR, a_FR, b_FR, scaled_m)
    z = y / 1.0e6  # Convert into millions

    return z


def RMSE(populationObservations, populationPredicted):
    residual = populationObservations - populationPredictedAfter
    dimension = populationObservations.getDimension()
    SS = 0.0
    for i in range(dimension):
        marginalResidual = residual[:, i].asPoint()
        SS += marginalResidual.normSquare()
    size = populationObservations.getSize()
    mse = SS / (dimension * size)
    rmse = np.sqrt(mse)
    return rmse


def plot_calibrated_prediction(
    sample_US,
    sample_FR,
    timeObservationsSample,
    populationPredictedBefore,
    populationPredictedAfter,
    cali_name="Calibration",
):
    palette = ot.DrawableImplementation_BuildDefaultPalette(2)

    graph2 = ot.Graph("", "Time (years)", "Population (Millions)", True, "topleft")
    # Observations
    cloud = ot.Cloud(sample_US)
    cloud.setColor(palette[0])
    cloud.setLegend("Observations US")
    graph2.add(cloud)

    cloud = ot.Cloud(sample_FR)
    cloud.setColor(palette[1])
    cloud.setLegend("Observations FR")
    graph2.add(cloud)

    # Predictions before calibration
    cloud = ot.Curve(timeObservationsSample, populationPredictedBefore[:, 0])
    cloud.setLegend("Before (US)")
    cloud.setColor(palette[0])
    cloud.setLineStyle("dashed")
    graph2.add(cloud)
    cloud = ot.Curve(timeObservationsSample, populationPredictedBefore[:, 1])
    cloud.setLegend("Before (FR)")
    cloud.setColor(palette[1])
    cloud.setLineStyle("dashed")
    graph2.add(cloud)

    # Predictions after calibration
    cloud = ot.Curve(timeObservationsSample, populationPredictedAfter[:, 0])
    cloud.setLegend("After (US)")
    cloud.setColor(palette[0])
    graph2.add(cloud)
    cloud = ot.Curve(timeObservationsSample, populationPredictedAfter[:, 1])
    cloud.setLegend("After (FR)")
    cloud.setColor(palette[1])
    graph2.add(cloud)

    rmse = RMSE(populationObservations, populationPredictedAfter)
    graph2.setTitle(
        "%s, $\hat{m}$ = %.2f mill., RMSE = %.1f" % (cali_name, thetaMAP[4], rmse)
    )
    return graph2


def plot_model_vs_data(sample_US, sample_FR, theta, model, timeObservations):

    palette = ot.DrawableImplementation_BuildDefaultPalette(2)

    graph2 = ot.Graph("", "Time (years)", "Population (Millions)", True, "topleft")
    # Observations
    cloud = ot.Cloud(sample_US)
    cloud.setLegend("Observations US")
    cloud.setColor(palette[0])
    graph2.add(cloud)

    cloud = ot.Cloud(sample_FR)
    cloud.setLegend("Observations FR")
    cloud.setColor(palette[1])
    graph2.add(cloud)

    # Predictions
    model.setParameter(theta)
    populationPredicted = model(timeObservations)
    cloud = ot.Curve(timeObservations, populationPredicted[:, 0])
    cloud.setLegend("Predictions US")
    cloud.setColor(palette[0])
    graph2.add(cloud)

    cloud = ot.Curve(timeObservations, populationPredicted[:, 1])
    cloud.setLegend("Predictions FR")
    cloud.setColor(palette[1])
    graph2.add(cloud)
    return graph2


def decorateParameterDistribution(grid):
    nbcols = grid.getNbColumns()
    for i in range(nbcols):
        graph = grid.getGraph(0, i)
        if i == nbcols - 1:
            graph.setLegends(["PDF", "Posterior", "Candidate"])
        else:
            graph.setLegends([""])
        if i > 0:
            graph.setYTitle("")
        grid.setGraph(0, i, graph)
    return grid

def decorateResiduals(grid):
    graph = grid.getGraph(0, 0)
    graph.setLegends([""])
    graph.setXTitle("$\epsilon_{US}$")
    graph.setTitle("")
    grid.setGraph(0, 0, graph)
    graph = grid.getGraph(0, 1)
    graph.setXTitle("$\epsilon_{FR}$")
    graph.setYTitle("")
    graph.setTitle("")
    graph.setLegends(["Initial", "Calibrated", "Obs. err."])
    grid.setGraph(0, 1, graph)
    return grid


def printRMSEValue(calibrationResult, thetaPrior):
    residualFunction = calibrationResult.getResidualFunction()
    residualsBefore = residualFunction(thetaPrior)
    dimension = residualFunction.getOutputDimension()
    rmseBefore = np.sqrt(residualsBefore.normSquare() / dimension)
    print("Before = %.2f" % (rmseBefore))
    thetaMAP = calibrationResult.getParameterMAP()
    residualsAfter = residualFunction(thetaMAP)
    rmseAfter = np.sqrt(residualsAfter.normSquare() / dimension)
    print("After = %.2f" % (rmseAfter))
    return None

data_US = np.array(
    [
        [1790, 3.9],
        [1800, 5.3],
        [1810, 7.2],
        [1820, 9.6],
        [1830, 13],
        [1840, 17],
        [1850, 23],
        [1860, 31],
        [1870, 39],
        [1880, 50],
        [1890, 62],
        [1900, 76],
        [1910, 92],
        [1920, 106],
        [1930, 123],
        [1940, 132],
        [1950, 151],
        [1960, 179],
        [1970, 203],
        [1980, 221],
        [1990, 250],
        [2000, 281],
    ]
)
sample_US = ot.Sample(data_US)

data_FR = np.array(
    [
        [1790, 28.1],
        [1800, 29.4],
        [1810, 30.0],
        [1820, 31.6],
        [1830, 33.6],
        [1840, 34.9],
        [1850, 36.5],
        [1860, 37.4],
        [1870, 37.7],
        [1880, 39.2],
        [1890, 39.9],
        [1900, 40.7],
        [1910, 41.4],
        [1920, 38.9],
        [1930, 41.3],
        [1940, 40.7],
        [1950, 41.6],
        [1960, 45.5],
        [1970, 50.5],
        [1980, 53.7],
        [1990, 56.6],
        [2000, 58.9],
    ]
)
sample_FR = ot.Sample(data_FR)
number_of_time_steps = sample_US.getSize()

# Get the times
timeObservationsSample = ot.Sample(sample_US[:, 0])

# Agregates the two samples

# Prior predictions
a_US = 0.03134
c_US = -22.58
a_FR = 0.005
c_FR = -40.0
m = 1.0
thetaPrior = ot.Point([a_US, c_US, a_FR, c_FR, m])
print("thetaPrior=", thetaPrior)
number_of_parameters = len(thetaPrior)

# Create the model
nb_data_set = 2
input_dimension = number_of_time_steps + number_of_parameters
output_dimension = number_of_time_steps * nb_data_set
logisticModelPy = ot.PythonFunction(6, 2, logisticModel)
model = ot.ParametricFunction(logisticModelPy, [1, 2, 3, 4, 5], thetaPrior)
model.setParameterDescription(["$a_{US}$", "$c_{US}$", "$a_{FR}$", "$c_{FR}$", "$m$"])

# There is a single observation.
# It agregates the US and FR populations.
data = np.vstack([data_US[:, 1], data_FR[:, 1]]).T
populationObservations = ot.Sample(data)

populationPredictedBefore = model(timeObservationsSample)

# Observed data
graph = ot.Graph("", "Time (years)", "Population (Millions)", True, "topleft")
cloud = ot.Cloud(sample_US)
cloud.setLegend("Observations US")
graph.add(cloud)
cloud = ot.Cloud(sample_FR)
cloud.setLegend("Observations FR")
graph.add(cloud)
graph.setColors(ot.DrawableImplementation_BuildDefaultPalette(2))
view = otv.View(graph, figure_kw={"figsize": (3, 2)})
view.save("logistique_horizontale_obs.pdf", bbox_inches="tight")

graph = plot_model_vs_data(
    sample_US,
    sample_FR,
    thetaPrior,
    model,
    timeObservationsSample,
)
graph.setTitle("Before calibration. $m$=%.1f" % (thetaPrior[4]))
view = otv.View(graph, figure_kw={"figsize": (4, 3)})
view.save("logistique_horizontale_before.pdf", bbox_inches="tight")

# -----------------------------------------------------------------------------
#
print("+ LinearLeastSquaresCalibration")

algo = ot.LinearLeastSquaresCalibration(
    model,
    timeObservationsSample,
    populationObservations,
    thetaPrior,
)
algo.run()
calibrationResult = algo.getResult()

thetaMAP = calibrationResult.getParameterMAP()
print("thetaMAP=", thetaMAP)

thetaPosterior = calibrationResult.getParameterPosterior()

model.setParameter(thetaMAP)
populationPredictedAfter = model(timeObservationsSample)

graph = plot_calibrated_prediction(
    sample_US,
    sample_FR,
    timeObservationsSample,
    populationPredictedBefore,
    populationPredictedAfter,
    cali_name="LLS",
)

view = otv.View(graph, figure_kw={"figsize": (4, 3)})
view.save("logistique_horizontale_LLSQ.pdf", bbox_inches="tight")

grid = calibrationResult.drawParameterDistributions()
grid = decorateParameterDistribution(grid)
view = otv.View(grid, figure_kw={"figsize": (12, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
pl.subplots_adjust(wspace=0.4)
_= pl.suptitle("LLS")
view.save("logistique_horizontale_LLSQ_theta.pdf", bbox_inches="tight")

observationError = calibrationResult.getObservationsError()
sigma_obs = observationError.getStandardDeviation()
print("sigma obs. =", sigma_obs)

observationError = calibrationResult.getObservationsError()
observationError.setDescription(["$\epsilon_{US}$", "$\epsilon_{FR}$"])
graph = observationError.drawPDF()
graph.setTitle("LLS: " + graph.getTitle())
view = otv.View(graph, figure_kw={"figsize": (3, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
view.save("logistique_horizontale_LLSQ_obserr.pdf", bbox_inches="tight")

grid = calibrationResult.drawResiduals()
grid = decorateResiduals(grid)
grid.setTitle("LLS: Residual analysis")
view = otv.View(grid, figure_kw={"figsize": (4, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
pl.subplots_adjust(wspace=0.4, top = 0.8)
view.save("logistique_horizontale_LLSQ_residuals.pdf", bbox_inches="tight")

print("LLSQ RMSE ")
printRMSEValue(calibrationResult, thetaPrior)

# -----------------------------------------------------------------------------
#
print("+ NonLinearLeastSquaresCalibration")

"""
If 

ot.ResourceMap_SetAsUnsignedInteger("NonLinearLeastSquaresCalibration-BootstrapSize", 0)

then an exception is generated:
    RuntimeError: InternalException : Error: the covariance of the posterior 
    distribution is not definite positive. 
    The problem may be not identifiable. 
    Try to increase the "LinearLeastSquaresCalibration-Regularization" key in 
    ResourceMap

The covariance of theta star is difficult to compute with Laplace approximation.
We use bootstrap instead.
"""
ot.ResourceMap_SetAsUnsignedInteger(
    "NonLinearLeastSquaresCalibration-BootstrapSize", 10
)
algo = ot.NonLinearLeastSquaresCalibration(
    model,
    timeObservationsSample,
    populationObservations,
    thetaPrior,
)
algo.run()
calibrationResult = algo.getResult()

thetaMAP = calibrationResult.getParameterMAP()
print("thetaMAP=", thetaMAP)

thetaPosterior = calibrationResult.getParameterPosterior()
model.setParameter(thetaMAP)
populationPredictedAfter = model(timeObservationsSample)

graph = plot_calibrated_prediction(
    sample_US,
    sample_FR,
    timeObservationsSample,
    populationPredictedBefore,
    populationPredictedAfter,
    cali_name="NLLS",
)
view = otv.View(graph, figure_kw={"figsize": (4, 3)})
view.save("logistique_horizontale_NLSQ.pdf", bbox_inches="tight")

# Fails
# graph = calibrationResult.drawParameterDistributions()
# otv.View(graph, figure_kw = {"figsize": (12, 3)})
# TypeError: InvalidArgumentException : Error: cannot draw a PDF with xMax <= xMin, here xmin=-73968.5 and xmax=-73968.5

observationError = calibrationResult.getObservationsError()
sigma_obs = observationError.getStandardDeviation()
print("sigma obs. =", sigma_obs)

observationError.setDescription(["$\epsilon_{US}$", "$\epsilon_{FR}$"])
graph = observationError.drawPDF()
graph.setTitle("NLLS: " + graph.getTitle())
view = otv.View(graph, figure_kw={"figsize": (3, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
view.save("logistique_horizontale_NLSQ_obserr.pdf", bbox_inches="tight")

grid = calibrationResult.drawResiduals()
grid = decorateResiduals(grid)
grid.setTitle("NLLS: Residual analysis")
view = otv.View(grid, figure_kw={"figsize": (4, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
pl.subplots_adjust(wspace=0.4, top = 0.8)
view.save("logistique_horizontale_NLSQ_residuals.pdf", bbox_inches="tight")

print("NLSQ RMSE ")
printRMSEValue(calibrationResult, thetaPrior)

# -----------------------------------------------------------------------------
#
print("+ GaussianLinearCalibration")

# Prior distribution
sigmaA_US = 0.1 * a_US
sigmaC_US = 10.0
sigmaA_FR = 0.1 * a_FR
sigmaC_FR = 10.0
sigmaM = 1.0
print("Prior Sigma = ", [sigmaA_US, sigmaC_US, sigmaA_FR, sigmaC_FR, sigmaM])

parameterCovariance = ot.CovarianceMatrix(5)
parameterCovariance[0, 0] = sigmaA_US ** 2
parameterCovariance[1, 1] = sigmaC_US ** 2
parameterCovariance[2, 2] = sigmaA_FR ** 2
parameterCovariance[3, 3] = sigmaC_FR ** 2
parameterCovariance[4, 4] = sigmaM ** 2

# Standard deviation of the population observation error
# We assume independent covariance errors with standard deviation
# error equal to 10.0 millions.
sigma_pop = 10.0
errorCovariance = ot.CovarianceMatrix(2)
errorCovariance[0, 0] = sigma_pop ** 2
errorCovariance[1, 1] = sigma_pop ** 2

algo = ot.GaussianLinearCalibration(
    model,
    timeObservationsSample,
    populationObservations,
    thetaPrior,
    parameterCovariance,
    errorCovariance,
)

algo.run()
calibrationResult = algo.getResult()

thetaMAP = calibrationResult.getParameterMAP()
print("thetaMAP=", thetaMAP)

thetaPosterior = calibrationResult.getParameterPosterior()

model.setParameter(thetaMAP)
populationPredictedAfter = model(timeObservationsSample)

graph = plot_calibrated_prediction(
    sample_US,
    sample_FR,
    timeObservationsSample,
    populationPredictedBefore,
    populationPredictedAfter,
    cali_name="Gaussian Lin.",
)
view = otv.View(graph, figure_kw={"figsize": (4, 3)})
view.save("logistique_horizontale_GL.pdf", bbox_inches="tight")

grid = calibrationResult.drawParameterDistributions()
grid = decorateParameterDistribution(grid)
view = otv.View(grid, figure_kw={"figsize": (12, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
pl.subplots_adjust(wspace=0.4)
_= pl.suptitle("GaussianLinearCalibration")
view.save("logistique_horizontale_GL_theta.pdf", bbox_inches="tight")

observationError = calibrationResult.getObservationsError()
observationError.setDescription(["$\epsilon_{US}$", "$\epsilon_{FR}$"])
graph = observationError.drawPDF()
graph.setTitle("Gaussian Lin.: " + graph.getTitle())
view = otv.View(graph, figure_kw={"figsize": (3, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
view.save("logistique_horizontale_GL_obserr.pdf", bbox_inches="tight")

grid = calibrationResult.drawResiduals()
grid = decorateResiduals(grid)
grid.setTitle("Gaussian Lin.: Residual analysis")
view = otv.View(grid, figure_kw={"figsize": (4, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
pl.subplots_adjust(wspace=0.4, top = 0.8)
view.save("logistique_horizontale_GL_residuals.pdf", bbox_inches="tight")

print("GL RMSE ")
printRMSEValue(calibrationResult, thetaPrior)

# -----------------------------------------------------------------------------
#
print("+ GaussianNonLinearCalibration")

algo = ot.GaussianNonLinearCalibration(
    model,
    timeObservationsSample,
    populationObservations,
    thetaPrior,
    parameterCovariance,
    errorCovariance,
)

algo.run()
calibrationResult = algo.getResult()

thetaMAP = calibrationResult.getParameterMAP()
print("thetaMAP=", thetaMAP)

thetaPosterior = calibrationResult.getParameterPosterior()

model.setParameter(thetaMAP)
populationPredictedAfter = model(timeObservationsSample)

graph = plot_calibrated_prediction(
    sample_US,
    sample_FR,
    timeObservationsSample,
    populationPredictedBefore,
    populationPredictedAfter,
    cali_name="Gaussian N.L.",
)
view = otv.View(graph, figure_kw={"figsize": (4, 3)})
view.save("logistique_horizontale_GNL.pdf", bbox_inches="tight")

grid = calibrationResult.drawParameterDistributions()
grid = decorateParameterDistribution(grid)
view = otv.View(grid, figure_kw={"figsize": (12, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
pl.subplots_adjust(wspace=0.4)
_= pl.suptitle("Gaussian N.L.")
view.save("logistique_horizontale_GNL_theta.pdf", bbox_inches="tight")

observationError = calibrationResult.getObservationsError()
observationError.setDescription(["$\epsilon_{US}$", "$\epsilon_{FR}$"])
graph = observationError.drawPDF()
graph.setTitle("Gaussian N.L.: " + graph.getTitle())
view = otv.View(graph, figure_kw={"figsize": (3, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
view.save("logistique_horizontale_GNL_obserr.pdf", bbox_inches="tight")

grid = calibrationResult.drawResiduals()
grid = decorateResiduals(grid)
grid.setTitle("Gaussian N.L.: Residual analysis")
view = otv.View(grid, figure_kw={"figsize": (4, 3)},
                legend_kw={"bbox_to_anchor":(1.0, 1.0), "loc":"upper left"})
pl.subplots_adjust(wspace=0.4, top = 0.8)
view.save("logistique_horizontale_GNL_residuals.pdf", bbox_inches="tight")

print("GNL RMSE ")
printRMSEValue(calibrationResult, thetaPrior)
