#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:31:26 2020
logistic calibration 2 countries
@author: d16574 - Sanaa Zannane. +Michaël Baudin (2021)

On considère un 1 échantillon de taille 1 en dimension 44.

Le modèle est :

y_US(t) = m + a_US * y0 / (b_US * y0 + (a_US - b_US * y0) * np.exp(-a_US * (t - t0)))
y_FR(t) = m + a_FR * y0 / (b_FR * y0 + (a_FR - b_FR * y0) * np.exp(-a_FR * (t - t0)))

Les 5 paramètres sont (a_US, b_US, a_FR, b_FR, m).

Les 11 dates observées sont dans l'intervalle [1790, 2000]. 
"""

import openturns as ot
import numpy as np
import openturns.viewer as otv

# Functions + data + constants


def modele_une_population(t, t0, y0, a, b, m):
    y = m + a * y0 / (b * y0 + (a - b * y0) * np.exp(-a * (t - t0)))
    return y


def logisticModel(X):
    t = [X[i] for i in range(number_of_time_steps)]
    a_US = X[number_of_time_steps]
    c_US = X[number_of_time_steps + 1]
    a_FR = X[number_of_time_steps + 2]
    c_FR = X[number_of_time_steps + 3]

    m = X[number_of_time_steps + 4]

    t0 = 1790.0
    y0_US = 3.9e6
    y0_FR = 28.1e6
    b_US = np.exp(c_US)
    b_FR = np.exp(c_FR)
    y = ot.Point(number_of_time_steps * 2)

    scaled_m = m * 1.0e6

    for i in range(number_of_time_steps):
        y[i] = modele_une_population(t[i], t0, y0_US, a_US, b_US, scaled_m)
        y[number_of_time_steps + i] = modele_une_population(
            t[i], t0, y0_FR, a_FR, b_FR, scaled_m
        )
    z = y / 1.0e6  # Convert into millions

    return z


def RMSE(populationObservations, populationPredicted):
    residual = populationObservations - populationPredictedAfter
    dimension = populationObservations.getDimension()
    mse = residual.normSquare() / dimension
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
    cloud.setLegend("Obs. US")
    graph2.add(cloud)

    cloud = ot.Cloud(sample_FR)
    cloud.setColor(palette[1])
    cloud.setLegend("Obs. FR")
    graph2.add(cloud)

    # Predictions before calibration
    number_of_time_steps = timeObservationsSample.getDimension()
    cloud = ot.Curve(
        timeObservationsSample, populationPredictedBefore[0:number_of_time_steps]
    )
    cloud.setLegend("Before (US)")
    cloud.setColor(palette[0])
    cloud.setLineStyle("dashed")
    graph2.add(cloud)
    cloud = ot.Curve(
        timeObservationsSample, populationPredictedBefore[number_of_time_steps:]
    )
    cloud.setLegend("Before (FR)")
    cloud.setColor(palette[1])
    cloud.setLineStyle("dashed")
    graph2.add(cloud)

    # Predictions after calibration
    cloud = ot.Curve(
        timeObservationsSample, populationPredictedAfter[0:number_of_time_steps]
    )
    cloud.setLegend("After (US)")
    cloud.setColor(palette[0])
    graph2.add(cloud)
    cloud = ot.Curve(
        timeObservationsSample, populationPredictedAfter[number_of_time_steps:]
    )
    cloud.setLegend("After (FR)")
    cloud.setColor(palette[1])
    graph2.add(cloud)

    rmse = RMSE(populationObservations[0], populationPredictedAfter)
    graph2.setTitle(
        "%s, $\hat{m}$ = %.1f mill., RMSE = %.1f"
        % (cali_name, thetaMAP[4] / 1.0e6, rmse)
    )
    return graph2


def plot_model_vs_data(sample_US, sample_FR, theta, model, timeObservations):
    palette = ot.DrawableImplementation_BuildDefaultPalette(2)
    graph2 = ot.Graph("", "Time (years)", "Population (Millions)", True, "topleft")
    # Observations
    cloud = ot.Cloud(sample_US)
    cloud.setLegend("Obs. US")
    cloud.setColor(palette[0])
    graph2.add(cloud)

    cloud = ot.Cloud(sample_FR)
    cloud.setLegend("Obs. FR")
    cloud.setColor(palette[1])
    graph2.add(cloud)

    # Predictions
    model.setParameter(theta)
    populationPredicted = model(timeObservations)
    number_of_time_steps = timeObservations.getDimension()
    cloud = ot.Curve(timeObservations, populationPredicted[0:number_of_time_steps])
    cloud.setLegend("Pred. US")
    cloud.setColor(palette[0])
    graph2.add(cloud)
    cloud = ot.Curve(timeObservations, populationPredicted[number_of_time_steps:])
    cloud.setLegend("Pred. FR")
    cloud.setColor(palette[1])
    graph2.add(cloud)
    return graph2

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

# Get the times
timeObservationsPoint = sample_US[:, 0].asPoint()
timeObservationsSample = ot.Sample([timeObservationsPoint])

# Agregates the two samples
observedSample = ot.Sample(data_US)
observedSample.add(data_FR)

# Prior predictions
a_US = 0.03134
c_US = -22.58
a_FR = 0.005
c_FR = -40.0
m = 1.0
thetaPrior = [a_US, c_US, a_FR, c_FR, m]
print("thetaPrior=", thetaPrior)
number_of_parameters = len(thetaPrior)

# Create the model
nb_data_set = 2
number_of_time_steps = len(data_US)
input_dimension = number_of_time_steps + number_of_parameters
output_dimension = number_of_time_steps * nb_data_set
logisticModelPy = ot.PythonFunction(
    input_dimension, number_of_time_steps * nb_data_set, logisticModel
)
model = ot.ParametricFunction(logisticModelPy, [22, 23, 24, 25, 26], thetaPrior)

# There is a single observation.
# It agregates the US and FR populations.
populationObservations = ot.Sample(1, output_dimension)
populationObservations[0] = observedSample[:, 1].asPoint()

populationPredictedBefore = model(timeObservationsPoint)

# Observed data
graph = ot.Graph("", "Time (years)", "Population (Millions)", True, "topleft")
cloud = ot.Cloud(sample_US)
cloud.setLegend("Obs. US")
graph.add(cloud)
cloud = ot.Cloud(sample_FR)
cloud.setLegend("Obs. FR")
graph.add(cloud)
graph.setColors(ot.DrawableImplementation_BuildDefaultPalette(2))
view = otv.View(graph, figure_kw={"figsize": (3, 2)})
view.save("logistique_verticale_obs.pdf", bbox_inches="tight")

graph = plot_model_vs_data(
    sample_US, sample_FR, thetaPrior, model, timeObservationsPoint
)
graph.setTitle("Before calibration. $m$=%.1f" % (thetaPrior[4]))
view = otv.View(graph, figure_kw={"figsize": (3, 2)})
view.save("logistique_verticale_before.pdf", bbox_inches="tight")

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
populationPredictedAfter = model(timeObservationsPoint)

graph = plot_calibrated_prediction(
    sample_US,
    sample_FR,
    timeObservationsPoint,
    populationPredictedBefore,
    populationPredictedAfter,
    cali_name="LLS",
)

view = otv.View(graph, figure_kw={"figsize": (4, 3)})
view.save("logistique_verticale_LLSQ.pdf", bbox_inches="tight")

print("LLSQ RMSE ")
printRMSEValue(calibrationResult, thetaPrior)

# -----------------------------------------------------------------------------
#
print("+ NonLinearLeastSquaresCalibration")

try:
    """
    An exception is generated:
        InvalidArgumentException : Error: cannot build a Normal distribution
        from a sample of size < 2

    This is because of :
        https://github.com/openturns/openturns/blob/91f7c99f9aae2f05ee746f2263beb2fb4928e07c/lib/src/Uncertainty/Bayesian/NonLinearLeastSquaresCalibration.cxx#L266

    corresponding to the code :
    const Normal error(NormalFactory().buildAsNormal(residual));

    This prevents from estimating the distribution of the residuals, because the
    sample size is equal to 1.

    The alternative provided in the linear least squares case is to create a
    Normal with an infinite variance:

        https://github.com/openturns/openturns/blob/91f7c99f9aae2f05ee746f2263beb2fb4928e07c/lib/src/Uncertainty/Bayesian/LinearLeastSquaresCalibration.cxx#L138
    """
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
    populationPredictedAfter = model(timeObservationsPoint)

    graph = plot_calibrated_prediction(
        sample_US,
        sample_FR,
        timeObservationsPoint,
        populationPredictedBefore,
        populationPredictedAfter,
        cali_name="NLLS",
    )

    view = otv.View(graph)
    view.save("logistique_verticale_NLSQ.pdf", bbox_inches="tight")

    print("NLSQ RMSE ")
    printRMSEValue(calibrationResult, thetaPrior)

except TypeError as inst:
    print(inst)


# -----------------------------------------------------------------------------
#
print("+ GaussianLinearCalibration")

# prior distribution
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
# error equal to 10 millions.
sigma_pop = 10.0
output_dimension = model.getOutputDimension()
errorCovariance = ot.CovarianceMatrix(output_dimension)
for i in range(output_dimension):
    errorCovariance[i, i] = sigma_pop ** 2

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
populationPredictedAfter = model(timeObservationsPoint)

graph = plot_calibrated_prediction(
    sample_US,
    sample_FR,
    timeObservationsPoint,
    populationPredictedBefore,
    populationPredictedAfter,
    cali_name="Gaussian Lin.",
)

view = otv.View(graph, figure_kw={"figsize": (4, 3)})
view.save("logistique_verticale_GL.pdf", bbox_inches="tight")

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
populationPredictedAfter = model(timeObservationsPoint)

graph = plot_calibrated_prediction(
    sample_US,
    sample_FR,
    timeObservationsPoint,
    populationPredictedBefore,
    populationPredictedAfter,
    cali_name="Gaussian NL",
)

view = otv.View(graph, figure_kw={"figsize": (4, 3)})
view.save("logistique_verticale_GNL.pdf", bbox_inches="tight")

print("GNL RMSE ")
printRMSEValue(calibrationResult, thetaPrior)
