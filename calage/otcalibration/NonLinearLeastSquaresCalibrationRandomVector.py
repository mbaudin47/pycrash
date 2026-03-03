#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample from the distribution of the estimator of non linear least squares.
"""

import openturns as ot


class NonLinearLeastSquaresCalibrationRandomVector(ot.PythonRandomVector):
    def __init__(
        self, model, trueParameter, inputObservations, observationOutputError, candidate
    ):
        """
        Create a non-linear least squares random vector.
        
        The outputs are generated from the true parameter values. 
        
        The input and output observations are constant. 

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
        """
        self.model = model
        self.trueParameter = trueParameter
        self.inputObservations = inputObservations
        sample_size = inputObservations.getSize()
        self.sample_size = sample_size
        self.observationOutputError = observationOutputError
        self.candidate = candidate
        # 1. Generate exact outputs
        self.model.setParameter(self.trueParameter)
        self.outputSample = self.model(self.inputObservations)
        dimension = trueParameter.getDimension()
        super(NonLinearLeastSquaresCalibrationRandomVector, self).__init__(dimension)

    def getRealization(self):
        """
        Return a realization of the random vector of the estimator.

        Returns
        -------
        parameter : ot.Point(parameterDimension)
            A realization of the estimator.
        """
        # 2. Add observation error
        outputError = self.observationOutputError.getSample(self.sample_size)
        outputObservations = self.outputSample + outputError
        # 3. Calibrate
        self.model.setParameter(self.candidate)  # Just to be sure
        algo = ot.NonLinearLeastSquaresCalibration(
            self.model,
            self.inputObservations,
            outputObservations,
            self.candidate,
        )
        algo.run()
        calibrationResult = algo.getResult()
        parameter = calibrationResult.getParameterMAP()
        return parameter
