#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objectif : le but de ce texte est d'observer le comportement du calage 
par moindres carrés linéaires lorsque le nombre d'observations augmente.

# Vérification des moindres carrés linéaires

## Introduction

L'objectif de ce document est de vérifier la distribution des paramètres et des résidus issus du problème de moindres carrés linéaires. On considère ici un modèle *exactement* linéaire. On considère un échantillon de taille 100.

On considère la classe `LinearLeastSquaresCalibration`. On considère le modèle linéaire :
$$
z = h(x, \boldsymbol{\theta}) = \theta_1 + \theta_2 x + \theta_3 x^2
$$
pour tout $x \in \mathbb{R}$ où le vecteur des paramètres est $\boldsymbol{\theta} = (\theta_1, \theta_2, \theta_3) \in \mathbb{R}^3$.

Les vrais paramètres sont 
$$
\boldsymbol{\theta}^\star = (12, 7, -8)^T.
$$

On considère $n=100$ observations. 
On considère les observations 
$$
y_i = h(x_i, \boldsymbol{\theta}) + \epsilon_i
$$
où $x_1, \ldots, x_n \in \mathbb{R}$ sont les entrées observées et $\epsilon_1, \ldots, \epsilon_n$ sont des réalisations d'une variable aléatoire Gaussienne :
$$
\epsilon \sim \mathcal{N}(0, \sigma^2)
$$
où $\sigma = 2$ est l'écart-type. 
Les entrées observées $x_1, \ldots, x_n$ sont disposées sur issues de $n$ réalisations indépendantes de la variable $X$ uniforme dans l'intervalle $[-1, 1]$. 

"""

import numpy as np
import openturns as ot
import openturns.viewer as otv
import tqdm as tqdm
import pylab as pl
import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "10"

ot.RandomGenerator.SetSeed(0)

# La cellule ci-dessous définit le modèle $h$.


def modelLineaire(X):
    x, theta1, theta2, theta3 = X
    y = theta1 + theta2 * x + theta3 * x ** 2
    return [y]


g = ot.PythonFunction(4, 1, modelLineaire)

# La variable `trueParameter` contient la valeur des paramètres "vrais".

trueParameter = ot.Point([12.0, 7.0, -8.0])

parameterDimension = trueParameter.getDimension()
parameterDimension

calibratedIndices = [1, 2, 3]
model = ot.ParametricFunction(g, calibratedIndices, trueParameter)

# On peut ensuite définir la table des entrées observées $(x_1, \ldots, x_n)$.

size = 20
data = np.linspace(-1.0, 1.0, size)
inputObservations = ot.Sample([[v] for v in data])

# On considère un bruit Gaussien de moyenne nulle et d'écart-type égal à 2.

outputObservationNoiseSigma = 2.0
observationOutputNoise = ot.Normal(0.0, outputObservationNoiseSigma)

# On peut alors produire les sorties du modèle ...

outputStress = model(inputObservations)

# ... et ajouter le bruit.

sampleNoiseH = observationOutputNoise.getSample(size)
outputObservations = outputStress + sampleNoiseH

# La variable `candidate` contient la valeur des paramètres de référence utilisés pour le calage.

candidate = ot.Point([8.0, 9.0, -6.0])

# On peut alors dessiner le modèle associé aux paramètres "vrais".

model.setParameter(trueParameter)
graph = model.draw(-1.0, 1.0)
graph.setLegends(["True"])
cloud = ot.Cloud(inputObservations, outputObservations)
cloud.setLegend("Observations")
graph.add(cloud)
model.setParameter(candidate)
curve = model.draw(-1.0, 1.0)
curve.setLegends(["Before"])
graph.add(curve)
graph.setColors(ot.Drawable_BuildDefaultPalette(3))
graph.setLegendPosition("bottomright")
otv.View(graph)

# On observe que le modèle avant calage est décalé par rapport aux observations. Cela implique que le calage a un travail à réaliser.

# Dans cette section, nous réalisons le calage en utilisant la classe `LinearLeastSquaresCalibration`. Pour être certain que le calage est réellement actif et que les paramètres "vrais" sont inconnus du calage, nous configurons les paramètres du calage par la méthode `setParameter` en utilisant les paramètres initiaux contenus dans la variable `candidate`.

model.setParameter(candidate)  # Met à jour le modèle
algo = ot.LinearLeastSquaresCalibration(
    model, inputObservations, outputObservations, candidate, "SVD"
)
algo.run()
calibrationResult = algo.getResult()
otv.View(calibrationResult.drawObservationsVsInputs())

# On observe que le modèle après calage est correctement au centre des observations : le calage a réduit les résidus.

# Observons les valeurs des paramètres après calage.

calibratedParameter = calibrationResult.getParameterMAP()
print(calibratedParameter)
"""
Rappelons que les vrais paramètres sont 
$$
\boldsymbol{\theta}^\star = (12, 7, -8)^T.
$$
On constate que le calage a produit des paramètres relativement proches des vrais paramètres. 
"""


def calibrate_model(size, trueParameter, model, candidate, observationOutputNoise):
    """
    Calibrate the model with linear least squares.

    Parameters
    ----------
    size : int
        La taille de l'échantillon.
    trueParameter : ot.Point
        Le paramètre vrai.
    model : ot.Function
        La fonction paramétrique.
    candidate : ot.Point
        Le point de référence.
    observationOutputNoise : ot.Normal
        La loi du résidu.

    Returns
    -------
    calibratedParameter : ot.Point
        Le vecteur des paramètres calés.

    """
    data = np.linspace(-1.0, 1.0, size)
    inputObservations = ot.Sample([[v] for v in data])

    model.setParameter(trueParameter)
    outputStress = model(inputObservations)
    sampleNoiseH = observationOutputNoise.getSample(size)
    outputObservations = outputStress + sampleNoiseH

    model.setParameter(candidate)  # Met à jour le modèle
    algo = ot.LinearLeastSquaresCalibration(
        model, inputObservations, outputObservations, candidate, "SVD"
    )
    algo.run()
    calibrationResult = algo.getResult()

    calibratedParameter = calibrationResult.getParameterMAP()
    return calibratedParameter


calibratedParameter = calibrate_model(
    size, trueParameter, model, candidate, observationOutputNoise
)
print(calibratedParameter)

# Convergence de l'estimateur
number_of_experiments = 100
log_size = np.logspace(1.1, 4.0, number_of_experiments)
size = 10
dimension = calibratedParameter.getDimension()
size_sample = ot.Sample(number_of_experiments, 1)
theta_sample = ot.Sample(number_of_experiments, dimension)
for i in tqdm.tqdm(range(number_of_experiments)):
    size = int(log_size[i])
    size_sample[i, 0] = size
    calibratedParameter = calibrate_model(
        size, trueParameter, model, candidate, observationOutputNoise
    )
    theta_sample[i] = calibratedParameter

# Dessin de la convergence de chaque composante
grid = ot.GridLayout(1, dimension)
for i in range(dimension):
    graph = ot.Graph("", "Sample size", "$\hat{\\theta}_{%d}$" % i, True, "topright")
    # 1. The sample of estimated parameters
    cloud = ot.Cloud(size_sample, theta_sample[:, i])
    graph.add(cloud)
    # 2. The sample of true parameters
    theta_true = ot.Sample([[trueParameter[i]] for j in range(number_of_experiments)])
    curve = ot.Curve(size_sample, theta_true)
    graph.add(curve)
    graph.setLogScale(ot.GraphImplementation.LOGX)
    graph.setColors(ot.Drawable_BuildTableauPalette(2))
    graph.setXTitle("$n$")
    grid.setGraph(0, i, graph)
grid.setTitle("")
view = otv.View(grid, figure_kw={"figsize": (5.0, 1.5)})
pl.subplots_adjust(wspace=0.6, top=0.8)
figure = view.getFigure()
figure.savefig("linear_calibration_LLSQ_theta.pdf", bbox_inches="tight")

# Calcule l'erreur absolue
absolute_error_sample = ot.Sample(number_of_experiments, dimension)
absolute_error_reference = ot.Sample(number_of_experiments, 1)
for i in range(number_of_experiments):
    absolute_error_reference[i, 0] = 1.0 / np.sqrt(size_sample[i, 0])
    for j in range(dimension):
        absolute_error_sample[i, j] = abs(theta_sample[i, j] - trueParameter[j])

# Dessin de la convergence de l'erreur absolue de chaque composante
graph = ot.Graph("", "Sample size", "$e_{abs}$", True, "topright")
curve = ot.Curve(size_sample, absolute_error_reference[:, 0])
curve.setLegend("$1/\sqrt{n}$")
curve.setLineStyle("dashed")
curve.setLineWidth(3.0)
graph.add(curve)
for i in range(dimension):
    cloud = ot.Cloud(size_sample, absolute_error_sample[:, i])
    cloud.setLegend("$\hat{\\theta}_{%d}$" % (i))
    cloud.setPointStyle("bullet")
    graph.add(cloud)
graph.setColors(ot.Drawable_BuildTableauPalette(1 + dimension))
graph.setLogScale(ot.GraphImplementation.LOGXY)
graph.setXTitle("$n$")
legends = graph.getLegends()
graph.setLegendPosition("")
view = otv.View(graph, figure_kw={"figsize": (3.0, 2.0)})
figure = view.getFigure()
figure.legend(legends, bbox_to_anchor=(1.3, 0.9))
figure.savefig("linear_calibration_LLSQ_eabs.pdf", bbox_inches="tight")
