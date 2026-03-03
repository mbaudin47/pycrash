#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample from the posterior distribution of linear 
Gaussian calibration.
"""

import openturns as ot


class LinearGaussianCalibrationRandomVector(ot.PythonRandomVector):
    def __init__(
        self,
        model,
        trueParameter,
        inputObservations,
        observationOutputError,
        candidate,
        parameterCovariance,
        errorCovariance,
    ):
        """
        Linear Gaussian random vector of the posterior parameter distribution.

        Parameter
        ---------
        model : ot.Function
            The parametric function to calibrate.
        trueParameter : ot.Point
            The true parameter value.
        inputObservations : ot.Sample
            The input sample.
        observationOutputError : ot.Distribution
            The true distribution of the observation error.
        candidate : ot.Point.
            The reference point where linearization takes place.
        parameterCovariance : ot.CovarianceMatrix
            The prior covariance of the Gaussian distribution of the parameter.
        errorCovariance : ot.CovarianceMatrix
            The covariance of the Gaussian distribution of the observation error.
        """
        self.model = model
        self.trueParameter = trueParameter
        self.inputObservations = inputObservations
        sample_size = inputObservations.getSize()
        self.sample_size = sample_size
        self.observationOutputError = observationOutputError
        self.candidate = candidate
        self.parameterCovariance = parameterCovariance
        self.errorCovariance = errorCovariance
        # 1. Generate exact outputs
        self.model.setParameter(self.trueParameter)
        self.outputSample = self.model(self.inputObservations)
        dimension = trueParameter.getDimension()
        super(LinearGaussianCalibrationRandomVector, self).__init__(dimension)

    def getRealization(self):
        """
        Return a realization of the random vector theta.

        Returns
        -------
        parameter : ot.Point(parameterDimension)
            A realization of the posterior random vector.
        """
        # 2. Add true observation error
        outputError = self.observationOutputError.getSample(self.sample_size)
        outputObservations = self.outputSample + outputError
        # 3. Calibrate
        algo = ot.GaussianLinearCalibration(
            self.model,
            self.inputObservations,
            outputObservations,
            self.candidate,
            self.parameterCovariance,
            self.errorCovariance,
        )
        algo.run()
        calibrationResult = algo.getResult()
        parameter = calibrationResult.getParameterMAP()
        return parameter

