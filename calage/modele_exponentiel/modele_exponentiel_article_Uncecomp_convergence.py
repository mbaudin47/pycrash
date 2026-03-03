#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On reproduit les calculs de l'article Uncecomp sur le modèle exponentiel.
On observe que le modèle ne converge pas vers la prétendue valeur 
"vraie".
C'est parceque le modèle n'est pas linéaire par rapport aux paramètres. 
"""

import openturns as ot
import openturns.viewer as otv
import otcalibration as otc
import numpy as np
import tqdm
import pylab as pl
import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "10"

size = 10
inputObservations = ot.Sample([[0.5 + i] for i in range(size)])

inVars = ["theta_1", "theta_2", "x"]
formulas = ["theta_1 + exp(theta_2 * x)"]
#formulas = ["theta_1 + theta_2 * x"]
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
model.setParameter(candidate)
curve = model.draw(0.5, 9.5)
curve.setLegends(["Candidate model"])
graph.add(curve)
graph.setLegendPosition("topleft")
graph.setColors(ot.Drawable_BuildDefaultPalette(3))
graph.setTitle("Exponential model.")
#
view = otv.View(graph, figure_kw={"figsize": (3.0, 2.0)})

def generate_data_and_model(size):
    input_regular_grid = np.linspace(0.5, 9.5, size)
    inputObservations = ot.Sample([[v] for v in input_regular_grid])
    
    inVars = ["theta_1", "theta_2", "x"]
    formulas = ["theta_1 + exp(theta_2 * x)"]
    #formulas = ["theta_1 + theta_2 * x"]
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
    
    # Set R
    errorCovariance = ot.CovarianceMatrix([[3.0]])
    
    candidate = ot.Point([1.0, 1.0])
    
    # Observations
    outputModel = model(inputObservations)
    observationErrorSample = observationOutputNoise.getSample(size)
    outputObservations = outputModel + observationErrorSample
    #
    result = [inputObservations, outputObservations, model, candidate, 
              parameter_true, parameterCovariance, errorCovariance]
    return result

size = 10
result = generate_data_and_model(size)
inputObservations, outputObservations, model, candidate, parameter_true, parameterCovariance, errorCovariance = result
    

def calibrate_model_with_Kalman(
    size,
):
    """
    Calibrate the model with Gaussian linear calibration.

    Parameters
    ----------
    size : int
        La taille de l'échantillon.

    Returns
    -------
    calibratedParameter : ot.Point
        Le vecteur des paramètres calés.

    """
    result = generate_data_and_model(size)
    inputObservations, outputObservations, model, candidate, parameter_true, parameterCovariance, errorCovariance = result

    thetaStarKalman, covarianceThetaStarKalman = otc.gaussianLinearCalibrationFromKalman(
        model,
        inputObservations,
        outputObservations,
        candidate,
        parameterCovariance,
        errorCovariance,
        verbose=False,
    )
    return thetaStarKalman

def calibrate_model_with_Cholesky(
    size,
):
    """
    Calibrate the model with Gaussian linear calibration.

    Parameters
    ----------
    size : int
        La taille de l'échantillon.

    Returns
    -------
    calibratedParameter : ot.Point
        Le vecteur des paramètres calés.

    """

    result = generate_data_and_model(size)
    inputObservations, outputObservations, model, candidate, parameter_true, parameterCovariance, errorCovariance = result

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
        verbose=False,
    )
    return thetaStarCholesky

size = 10
thetaStarKalman = calibrate_model_with_Kalman(size)
print("Theta (Kalman)=", thetaStarKalman)
thetaStarCholesky = calibrate_model_with_Cholesky(size)
print("Theta (Cholesky)=", thetaStarCholesky)

# Convergence de l'estimateur
number_of_experiments = 50
log_size = np.logspace(1.0, 3.2, number_of_experiments)
dimension = 2
size_sample = ot.Sample(number_of_experiments, 1)
theta_sample_Kalman = ot.Sample(number_of_experiments, dimension)
theta_sample_Cholesky = ot.Sample(number_of_experiments, dimension)
for i in tqdm.tqdm(range(number_of_experiments)):
    size = int(log_size[i])
    size_sample[i, 0] = size
    theta_sample_Kalman[i] = calibrate_model_with_Kalman(size)
    theta_sample_Cholesky[i] = calibrate_model_with_Cholesky(size)

# Dessin de la convergence de chaque composante
grid = ot.GridLayout(1, dimension)
for i in range(dimension):
    graph = ot.Graph("", "Sample size", "$\hat{\\theta}_{%d}$" % (1 + i), True, "topright")
    # 1. The sample of estimated parameters
    cloud = ot.Cloud(size_sample, theta_sample_Kalman[:, i])
    cloud.setLegend("K")
    graph.add(cloud)
    # 2. The sample of estimated parameters
    cloud = ot.Cloud(size_sample, theta_sample_Cholesky[:, i])
    cloud.setLegend("C")
    graph.add(cloud)
    # 3. The sample of true parameters
    theta_true = ot.Sample([[parameter_true[i]] for j in range(number_of_experiments)])
    curve = ot.Curve(size_sample, theta_true)
    graph.add(curve)
    graph.setLogScale(ot.GraphImplementation.LOGX)
    graph.setColors(ot.Drawable_BuildTableauPalette(3))
    graph.setXTitle("$n$")
    graph.setLegendPosition("topright")
    grid.setGraph(0, i, graph)
grid.setTitle("")
for i in range(dimension):
    graph = grid.getGraph(0, i)
    legends = graph.getLegends()
    graph.setLegendPosition("")
    grid.setGraph(0, i, graph)
view = otv.View(grid, figure_kw={"figsize": (3.0, 1.5)})
pl.subplots_adjust(wspace=0.6, top=0.8)
figure = view.getFigure()
figure.legend(legends, bbox_to_anchor=(1.1, 0.9))
figure.savefig("exponentiel_convergence.pdf", bbox_inches='tight')
