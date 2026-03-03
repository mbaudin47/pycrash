#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trefethen, Lloyd N., and David Bau III. Numerical linear algebra. Vol. 50. Siam, 1997.

Lecture 19. Stability of least squares algorithms. p.137
"""

import openturns as ot
import numpy as np
import pylab as pl

class LinearLeastSquaresValidation:
    def __init__(self, model, inputObservations, outputObservations, candidate, methodName="SVD"):
        self.model = model
        self.inputObservations = inputObservations
        self.outputObservations = outputObservations
        self.candidate = candidate
        self.methodName = methodName
        self.algo = ot.LinearLeastSquaresCalibration(
            model, inputObservations, outputObservations, candidate, methodName
        )
    def run(self):
        self.algo.run()
        self.calibrationResult = self.algo.getResult()
        # Calcule theta
        parameterEstimate = self.calibrationResult.getParameterMAP()
        residualFunction = self.calibrationResult.getResidualFunction()
        self.residuals = residualFunction(parameterEstimate)
        sin_theta = np.linalg.norm(self.residuals) / np.linalg.norm(self.outputObservations)
        self.theta = np.arcsin(sin_theta)
        # Calcul de la matrice de conception
        gradientObservations = self.algo.getGradientObservations()
        # Compute eta
        # eta = ||A|| * ||theta|| / ||A * theta||
        self.model.setParameter(parameterEstimate)
        self.predictions = self.model(self.inputObservations)
        self.eta = np.linalg.norm(gradientObservations) * np.linalg.norm(parameterEstimate) / np.linalg.norm(self.predictions)
        # Condition number of the design matrix
        self.kappa = np.linalg.cond(gradientObservations)

    def getResult(self):
        return self.calibrationResult

    def plotProjection(self, theta_horizontal_factor = 0.2, theta_vertical_factor = 0.1, 
                       figsize=(3.0, 2.0)):
        pl.figure(figsize=figsize)
        pl.plot([0.0, np.linalg.norm(self.predictions)], [0.0, 0.0], "x-", label="Prédictions")
        pl.plot([np.linalg.norm(self.predictions)] * 2, [0.0, np.linalg.norm(self.residuals)], "x-", label="Résidus")
        pl.plot([0.0, np.linalg.norm(self.predictions)], [0.0, np.linalg.norm(self.residuals)], "x-", label="Observations")
        pl.xlabel("$\\mathcal{R}(A)$")
        pl.ylabel("$\\mathbf{r}$")
        pl.text(np.linalg.norm(self.predictions) * theta_horizontal_factor, np.linalg.norm(self.residuals) * theta_vertical_factor, "$\\theta$")
        pl.title("$\\theta=$ %.3e (rad)" % (self.theta))
        pl.legend(bbox_to_anchor=(1.0, 1.0))
