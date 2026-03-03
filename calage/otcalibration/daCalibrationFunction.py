# -*- coding: utf-8 -*-
# Copyright (C) 2018 - 2019 - Michael Baudin

import openturns as ot

"""
Définit une fonction d'observation.
"""


class CalibrationXDimensionError(Exception):
    def __init__(self, inputNumberXObservations, expectedNumberXObservations):
        self.inputNumberXObservations = inputNumberXObservations
        self.expectedNumberXObservations = expectedNumberXObservations

    def __str__(self):
        return "Actual number of X observations is %d while expected is %d" % (
            self.inputNumberXObservations,
            self.expectedNumberXObservations,
        )


class DaCalibrationFunction(ot.OpenTURNSPythonFunction):
    def __init__(
        self,
        modelFunction,
        XobservationsIndices,
        calibratedIndices,
        numberXobservations,
    ):
        # La colonne i dans X correspond à l'entrée observationsIndices[i] dans modelFunction
        # La composante i dans theta correspond à l'entrée calibratedIndices[i] dans modelFunction
        self.Xobservations = None
        self.modelFunction = modelFunction
        self.XobservationsIndices = XobservationsIndices
        self.calibratedIndices = calibratedIndices
        self.numberXobservations = numberXobservations
        dimTheta = len(XobservationsIndices) + len(calibratedIndices)
        dimensionSingleY = self.modelFunction.getOutputDimension()
        dimensionPredictions = dimensionSingleY * numberXobservations
        self.dimensionPredictions = dimensionPredictions
        super(DaCalibrationFunction, self).__init__(dimTheta, dimensionPredictions)

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
        if self.numberXobservations != self.Xobservations.getSize():
            CalibrationXDimensionError(
                self.Xobservations.getSize(), self.numberXobservations
            )
        dimCalage = len(theta)
        dimXobservations = self.Xobservations.getDimension()
        dimEntree = dimCalage + dimXobservations
        dimSingleY = self.modelFunction.getOutputDimension()
        # Evaluation de la sortie du modèle
        fullOutput = ot.Sample(self.numberXobservations, dimSingleY)
        for i in range(self.numberXobservations):
            # Crée le vecteur des entrées
            entree = ot.Point(dimEntree)
            # D'abord, les entrées lues dans les données X observées
            for j in range(len(self.XobservationsIndices)):
                k = self.XobservationsIndices[j]
                entree[k] = self.Xobservations[i, j]
            # Puis les paramètres
            for j in range(dimCalage):
                k = self.calibratedIndices[j]
                entree[k] = theta[j]
            # Evalue la sortie
            fullOutput[i, :] = self.modelFunction(entree)
        # Compacte les sorties pour former un vecteur de prédictions
        sortie = ot.Point(self.dimensionPredictions, 1)
        k = 0
        for j in range(dimSingleY):
            sortie[k : k + self.numberXobservations] = fullOutput[:, j].asPoint()
            k = k + self.numberXobservations
        return sortie

    def setXobservations(self, Xobservations):
        # Configure the inputs
        self.Xobservations = Xobservations
        return None

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
        # Définit la fonction d'observation
        def obsFunctionForGradient(theta):
            sortie = self.obsFunction(theta)
            return sortie

        dimCalage = len(self.calibratedIndices)
        obsPyFunc = ot.PythonFunction(
            dimCalage, self.dimensionPredictions, obsFunctionForGradient
        )
        g = obsPyFunc.gradient(theta)
        return g
