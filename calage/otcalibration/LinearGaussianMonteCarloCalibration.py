#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a linear Gaussian Monte-Carlo calibration algorithm.
"""

import openturns as ot


class LinearGaussianMonteCarloCalibration:
    def __init__(
        self,
        model,
        inputObservations,
        outputObservations,
        parameterPrior,
        reSampleSize=1000,
        probabilityThreshold=0.1,
    ):
        """
        Create a linear Gaussian Monte-Carlo calibration algorithm.

        Parameters
        ----------
        parameterPrior: ot.Distribution
            The prior distribution of the parameter.
        reSampleSize : int
            The re-sample size.
        probabilityThreshold : float in [0, 1]
            The probability threshold for calibration.
        """
        self.model = model
        self.inputObservations = inputObservations
        self.outputObservations = outputObservations
        self.parameterPrior = parameterPrior
        self.reSampleSize = reSampleSize
        self.probabilityThreshold = probabilityThreshold

    def run(self):
        """
        Compute the Monte-Carlo solution.

        Repeat the following experiment reSampleSize times.
        * Generate a realization of the parameter vector from prior distribution.
        * Compute output predictions from input observations.
        * Compute the residual and its euclidian norm.

        We approximate the distribution of the residuals by kernel
        smoothing.
        We compute a threshold for the residual euclidian norm
        as a quantile of the distribution of the euclidian norms of
        the residuals.
        We select the parameter values which residual norm is lower than a
        threshold.
        We approximate the posterior distribution of the parameter with
        kernel smoothing.

        Returns
        -------
        parameterPosterior : ot.Distribution
            The posterior distribution of the parameter.
        """
        parameterDimension = self.model.getParameterDimension()
        parameterSample = ot.Sample(self.reSampleSize, parameterDimension)
        parameterSample.setDescription(self.model.getParameterDescription())
        sampleDistance = ot.Sample(self.reSampleSize, 1)
        for i in range(self.reSampleSize):
            parameterSample[i] = self.parameterPrior.getRealization()
            self.model.setParameter(parameterSample[i])
            predictions = self.model(self.inputObservations)
            residual = predictions.asPoint() - self.outputObservations.asPoint()
            sampleDistance[i, 0] = residual.norm()
        # Compute distribution of distances
        self.distanceDistribution = ot.KernelSmoothing().build(sampleDistance)
        self.distanceThreshold = self.distanceDistribution.computeQuantile(
            self.probabilityThreshold
        )[0]
        # Compute theta points achieving the criteria
        conditionalSampleIndex = 0  # Number of Theta achieving the criteria
        for i in range(self.reSampleSize):
            if sampleDistance[i, 0] < self.distanceThreshold:
                conditionalSampleIndex += 1
        conditionalSampleSize = conditionalSampleIndex
        # Create sample of conditional parameters
        conditionalSampleIndex = 0  # Number of Theta achieving the criteria
        self.conditionalParameterSample = ot.Sample(
            conditionalSampleSize, parameterDimension
        )
        for i in range(self.reSampleSize):
            if sampleDistance[i, 0] < self.distanceThreshold:
                self.conditionalParameterSample[
                    conditionalSampleIndex
                ] = parameterSample[i]
                conditionalSampleIndex += 1
        # Create distribution of parameters
        self.parameterPosterior = ot.KernelSmoothing().build(
            self.conditionalParameterSample
        )
        return self.parameterPosterior
