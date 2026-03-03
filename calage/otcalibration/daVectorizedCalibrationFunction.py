# -*- coding: utf-8 -*-
# Copyright (C) 2018 - 2019 - Michael Baudin

import openturns as ot

"""
Définit une fonction d'observation vectorisée.
"""


class DaVectorizedCalibrationFunction(ot.OpenTURNSPythonFunction):
    def __init__(self, modelFunction):
        # La colonne i dans X correspond à l'entrée observationsIndices[i] dans modelFunction
        # La composante i dans theta correspond à l'entrée calibratedIndices[i] dans modelFunction
        self.modelFunction = modelFunction
        self.dimTheta = modelFunction.getInputDimension()
        self.dimensionPredictions = modelFunction.getOutputDimension()
        super(DaVectorizedCalibrationFunction, self).__init__(
            self.dimTheta, self.dimensionPredictions
        )

    def _exec(self, theta):
        return self.obsFunction(theta)

    # Définit la fonction d'observation
    def obsFunction(self, theta):
        """
        Evaluates the calibration function.

        Parameters
        ----------
        theta : a Point

        Returns
        -------
        y : a Point, the value of the function
        """
        sortie = self.modelFunction(theta)
        return sortie

    # Définit la fonction d'observation
    def gradient(self, theta):
        """
        Evaluates the gradient of the calibration function.

        Parameters
        ----------
        theta : a Point

        Returns
        -------
        gradient : a Matrix
        """
        g = self.modelFunction.gradient(theta)
        return g
